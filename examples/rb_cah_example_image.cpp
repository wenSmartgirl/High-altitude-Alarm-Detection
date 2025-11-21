
#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include "depthai/depthai.hpp"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "ntkgf.h"
#include "rb_cah_example_utils.h"
#include "rb_cah_detector.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <mutex> // 互斥锁
#include <cstring>




using namespace std;
constexpr auto FPS = 10.0; 
bool SPARSE=false;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
const std::vector<std::string> kClassNames = {
    "normal",
    "RopeNoHelmet",
    "both",
    "HelmetNoRope"
};

struct PCBuffer {
    std::unique_ptr<dai::Point3f[]> data; 
    size_t vaild = 0;
    int w=0,h=0,cap=0;
    std::atomic<uint64_t> seq{0}; //帧序号 不可打断
    std::mutex mtx; //互斥锁
} gPC;

// 确保容量 尺寸变化重新分配
inline void ensurePCCapacity(int w, int h) {
    int need = w * h;
    if(need <= 0 ) return;
    if(need > gPC.cap){
        std::lock_guard<std::mutex> lk(gPC.mtx);
        gPC.data.reset(new dai::Point3f[need]); //new 分配
        gPC.cap = need;
        gPC.w = w;
        gPC.h = h;
    }else{
        gPC.w = w;
        gPC.h = h;
    }
}

// 复制messageGroup点云数据到连续内存
// 输入一帧点云  此帧宽度 高度
inline void copyPointCloudToBuffer(const std::vector<dai::Point3f>& pts, int w, int h) {
    ensurePCCapacity(w, h);
    const size_t need = static_cast<size_t>(w)*static_cast<size_t>(h);
    const size_t ncopy = std::min(pts.size(), need);
    // 临界区
    {
        std::lock_guard<std::mutex> lk(gPC.mtx);
        if(ncopy > 0 ){
            std::memcpy(gPC.data.get(), pts.data(), sizeof(dai::Point3f) * ncopy);
        }
        gPC.vaild = ncopy;
        gPC.w = w;gPC.h = h;
        gPC.seq.fetch_add(1, std::memory_order_release); //原子操作 帧序号+1
    }
}



//去噪、滤波、降采样、离群点剔除和法线估计，
//并最终输出处理后的点云以及其在空间中的最小/最大坐标
// threshold 控制统计滤波的邻域点数   max_points 控制最大点数
void Process_PointCloud(pcl::PointCloud<PointT>::Ptr& cloud,
                        PointT &min_point, PointT &max_point,
                        float voxel_size, int threshold, int max_points)
{
    if(!cloud || cloud->empty()){
        min_point = PointT(); max_point = PointT();//为空 初始化为原点返回
        return;
    }

    // 去NaN点
    pcl::PointCloud<PointT>::Ptr cloud_clean(new pcl::PointCloud<PointT>());
    std::vector<int> dummy; //存储被保留在原始点云中的点的索引
    pcl::removeNaNFromPointCloud(*cloud, *cloud_clean, dummy);
    if(cloud_clean->size() < 10){ // 清理后点数过少 原始数据差 写回干净点云
        cloud->swap(*cloud_clean); 
        pcl::getMinMax3D(*cloud, min_point, max_point);
        return;
    }
    // 直通滤波
    {
        pcl::PassThrough<PointT> pass; // 用于按某一坐标轴范围进行过滤
        pass.setInputCloud(cloud_clean);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(200.0f, 20000.0f); 
        pass.filter(*cloud_clean);
    }
    //wwdebug
    // 估计点云大致距离
    float dist = 0.0f;
    if(!cloud_clean->empty()){
        std::vector<float> zs;
        zs.reserve(cloud_clean->size());
        for(const auto& pt : cloud_clean->points){
            if(std::isfinite(pt.z)&& pt.z > 0.0f){
                zs.push_back(pt.z);
            }
        }
        if(!zs.empty()){
            std::nth_element(zs.begin(), zs.begin() + zs.size()/2, zs.end());
            float z_median = zs[zs.size()/2];
            dist = z_median*0.001f; // 转为米
        }
    }
    if(cloud_clean->size() < 10){
        cloud->swap(*cloud_clean);
        pcl::getMinMax3D(*cloud, min_point, max_point);
        return;
    }

    // 体素降采样
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud_clean);

    //wwdebug 根据距离调整体素大小
    float leaf_mm = voxel_size;//叶节点尺寸
    if(dist > 0.0f){
        dist = std::max(0.3f,std::min(dist,20.0f));
        if( dist < 7.0f ){
            leaf_mm = std::max(10.0f, voxel_size * 0.5f); 
        }else if( dist < 12.0f ){
            leaf_mm = std::max(20.0f, voxel_size * 0.8f);
        }else{
            leaf_mm = std::max(30.0f, voxel_size * 1.0f);
        }
    }
        

    float leaf = std::max(leaf_mm, 1.0f); // 防止填 0
    vg.setLeafSize(leaf, leaf, leaf);       
    pcl::PointCloud<PointT>::Ptr cloud_down(new pcl::PointCloud<PointT>());
    vg.filter(*cloud_down);

    // 点数上限
    if(max_points > 0 && cloud_down->size() > static_cast<size_t>(max_points)){
        cloud_down->points.resize(max_points);
        cloud_down->width  = max_points;
        cloud_down->height = 1;
    }

    // 统计离群点剔除
    // threshold 用作邻居数
    if(cloud_down->size() >= static_cast<size_t>(std::max(50, threshold))){
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud_down);
        sor.setMeanK(std::max(10, threshold)); 
        sor.setStddevMulThresh(1.0);           // 若某点的平均距离超过“全局平均距离 + 1倍标准差”，则剔除。
        pcl::PointCloud<PointT>::Ptr cloud_denoised(new pcl::PointCloud<PointT>());
        sor.filter(*cloud_denoised);
        cloud_down.swap(cloud_denoised);
    }


    // // 法线估计   需要足够多的邻居点
    // if(cloud_down->size() >= 30){
    //     pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    //     tree->setInputCloud(cloud_down); // 当前点云构建成 KdTree
    //     pcl::NormalEstimation<PointT, pcl::Normal> ne;
    //     ne.setInputCloud(cloud_down); 
    //     ne.setSearchMethod(tree);// 使用 KdTree 查找邻居
    //     ne.setKSearch(16); // 使用最近的16个点来拟合平面
    //     pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    //     ne.compute(*normals);
    // }

    //  在滤波之后计算 min/max
    pcl::getMinMax3D(*cloud_down, min_point, max_point);

    // 更新cloud
    cloud->swap(*cloud_down);
}

// 计算单个目标的稳定高度
bool compute_height(const dai::Point3f* points_ptr,int width,int height,
                    cv::Mat& redst,const RB_CAH_Ped_S& stTmp,bool DEBUG_DRAW,
                    float & stable_height,float & cand_height,int & cand_count,
                    float& out_height_m)
{
     // 手写 clamp  限制原始值 最小值 最大值 返回值 v--[lo,hi]
    auto clampi = [](float v, float lo, float hi)->float {
        return std::max(lo, std::min(v, hi));
    };
    auto median = [](const std::vector<float>& v)->float {
        std::vector<float> tmp = v;
        if(tmp.empty()) return std::numeric_limits<float>::quiet_NaN();
        // 快速找中位数
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        return tmp[tmp.size()/2];// 返回中位数
    };
    auto percentile = [](const std::vector<float>& v, float q)->float {
        std::vector<float> tmp = v;
        if(tmp.empty()) return std::numeric_limits<float>::quiet_NaN();
        size_t k = static_cast<size_t>(std::floor(q * (tmp.size() - 1)));
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        return tmp[k];
    };

    auto valid = [&](const dai::Point3f& p)->bool {
        // 有效深度范围
        const float Z_MIN = 10.0f;     
        const float Z_MAX = 30000.0f; 
        return std::isfinite(p.y) && std::isfinite(p.z) && p.z >= Z_MIN && p.z <= Z_MAX;
    };

    int u0 = stTmp.stPedBoundingBox.stTopLeft.s32X;
    int v0 = stTmp.stPedBoundingBox.stTopLeft.s32Y;
    int u1 = stTmp.stPedBoundingBox.stBottomRight.s32X;
    int v1 = stTmp.stPedBoundingBox.stBottomRight.s32Y;
    u0 = clampi(u0, 0, static_cast<int>(width) - 1);
    u1 = clampi(u1, 0, static_cast<int>(width) - 1);
    v0 = clampi(v0, 0, static_cast<int>(height) - 1);
    v1 = clampi(v1, 0, static_cast<int>(height) - 1);

    // 脚带
    int xL = u0 + (u1 - u0) * 2 / 10; 
    int xR = u0 + (u1 - u0) * 8 / 10; 
    xL = clampi(xL, u0, u1);
    xR = clampi(xR, u0, u1); 
    int boxHeight = v1 - v0 + 1;
    int footBandH = std::max(10,boxHeight/8);
    int footCenter = v1 + std::min(6,boxHeight/15);
    int footTop = clampi(footCenter-footBandH/2,v0,(int)height-1);
    int footBot = clampi(footCenter+footBandH/2,v0,(int)height-1);
    if(DEBUG_DRAW){
        cv::rectangle(redst,
            cv::Rect(cv::Point(xL,footTop),
            cv::Point(xR,std::min(footBot,(int)height-1))),
            cv::Scalar(0, 255, 255), 2);//黄色
        cv::putText(redst, "FOOT BAND", 
            cv::Point(xL, footTop - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
            
    std::vector<float> footY, footZ;
    footY.reserve((xR - xL + 1) * std::max(1, footBot - footTop + 1));
    footZ.reserve(footY.capacity());
    for(int yy = footTop; yy <= footBot; ++yy){
        int row = yy * static_cast<int>(width);
        for(int xx = xL; xx <= xR; ++xx){
            size_t idx = static_cast<size_t>(row + xx);
            if(idx >= gPC.vaild) continue;
            const auto &p = points_ptr[idx];  
            if(valid(p)) { footY.push_back(p.y); footZ.push_back(p.z); }
        }
    }

    if((int)footY.size() < 2) return false;
    float zFoot = median(footZ);   
    float dist = zFoot * 0.001f; 
    // z异常过滤 防止光污染
    const float MAX_VALID_DEPTH = 19.0f;
    if(dist > MAX_VALID_DEPTH) return false;
    float dist_m = std::max(0.3f,std::min(dist,20.0f));

    //过滤Z
    float DZ_FOOT_MM = 80.0f; 
    if(dist_m >= 7.0f && dist_m <= 12.0f) DZ_FOOT_MM = 110.0f; 
    else if(dist_m > 12.0f) DZ_FOOT_MM = 130.0f; 
    
    std::vector<float> footY_filt; footY_filt.reserve(footY.size());
    for(size_t k = 0; k < footY.size(); ++k){
        if(std::fabs(footZ[k] - zFoot) <= DZ_FOOT_MM) footY_filt.push_back(footY[k]);
    }
    if(static_cast<int>(footY_filt.size()) < 2) return false;
    std::cout << "footY_filt.size()=" << footY_filt.size() << std::endl;

    float yFoot = percentile(footY_filt, 0.85f); 
    std::cout<<"zfoot="<<zFoot<<std::endl;
    // 近地面的点 和 脚的深度相差不超过
    // 近地面 120mm以内 远地面往后+40mm-300mm 上限900mm 或 深度的4%
    float Z_NEAR_MIN = 0.0f;
    float Z_NEAR_MAX = 0.0f;
    float DZ_FAR_MIN = 0.0f;
    float DZ_FAR_MAX = 0.0f;
    int gTop = std::min(v1+5,(int)height-1);
    int gBot = std::min(v1+45,(int)height-1);

    if(dist_m >= 0.0f && dist_m <= 5.0f){
        Z_NEAR_MIN = 120.0f;
        Z_NEAR_MAX = 240.0f;
        DZ_FAR_MIN = 300.0f;
        DZ_FAR_MAX = 900.0f;
        gBot = std::min(v1 + 200, (int)height - 1);
    }else if(dist_m <= 9.0f){
        Z_NEAR_MIN = 220.0f;
        Z_NEAR_MAX = 420.0f;
        DZ_FAR_MIN = 320.0f;
        DZ_FAR_MAX = 1100.0f;
        gBot = std::min(v1 + 180, (int)height - 1);
    }else if( dist_m <= 14.0f){
        Z_NEAR_MIN = 260.0f;
        Z_NEAR_MAX = 520.0f;
        DZ_FAR_MIN = 360.0f;
        DZ_FAR_MAX = 1200.0f;
        gBot = std::min(v1 + 180, (int)height - 1);
    }else{
        Z_NEAR_MIN = 300.0f;
        Z_NEAR_MAX = 600.0f;
        DZ_FAR_MIN = 400.0f;
        DZ_FAR_MAX = 1300.0f;
        gBot = std::min(v1 + 140, (int)height - 1);
    }
    
    if (DEBUG_DRAW) {
        cv::rectangle(redst,cv::Rect(cv::Point(xL, gTop),cv::Point(xR, gBot)),
            cv::Scalar(0, 165, 255), 2 );
        cv::putText(redst, "GROUND SEARCH",cv::Point(xL, std::max(gTop - 5, 0)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,165,255), 1);
    }

    std::vector<float> gY_near; gY_near.reserve((xR - xL + 1) * std::max(1, gBot - gTop + 1));
    std::vector<float> gY_far;gY_far.reserve((xR - xL + 1) * std::max(1, gBot - gTop + 1));
    for(int yy = gTop;yy <=gBot;++yy){
        int row = yy * static_cast<int>(width);
        for(int xx = xL;xx <= xR;++xx){
            size_t idx = static_cast<size_t>(row + xx);
            if(idx >= gPC.vaild) continue;
            const auto &p = points_ptr[idx];
            if(std::isnan(p.z)||p.z<10.0f||p.z>30000.0f) continue;
            float dz = std::fabs(p.z - zFoot);
            if(dz >= Z_NEAR_MIN && dz <= Z_NEAR_MAX){
                gY_near.push_back(p.y);
                if(DEBUG_DRAW) redst.at<cv::Vec3b>(yy, xx) = cv::Vec3b(0,255,0);   // 近地面点：绿
            }else if(dz >= DZ_FAR_MIN && dz <= DZ_FAR_MAX){
                gY_far.push_back(p.y);
                if(DEBUG_DRAW) redst.at<cv::Vec3b>(yy, xx) = cv::Vec3b(255,0,0);   // 远地面点：蓝
            }    
        }
    }
    if(dist_m <3.0f && !gY_near.empty()) gY_far.clear();

    int MIN_CNT = 3;
    if(dist_m >= 7.0f && dist_m <= 12.0f) MIN_CNT = 2;
    else if (dist_m > 12.0f) MIN_CNT = 1;
    
    bool have_near = (static_cast<int>(gY_near.size()) >= MIN_CNT);
    bool have_far  = (static_cast<int>(gY_far.size())  >= MIN_CNT);

    float height_m = -1.0f;
    bool  got_height = false;
    constexpr float TO_M = 1.0f / 1000.0f;

    float h_near_m = -1.0f, h_far_m = -1.0f;
    float yGround_near = NAN,yGround_far = NAN;
    if(have_near){
        got_height = true;
        yGround_near = percentile(gY_near, 0.10f);  
        h_near_m = std::fabs(yFoot - yGround_near) * TO_M; 
        // std::cout <<"[GROUND] yFoot: " << yFoot
        //         << " yGround_near: " << yGround_near
        //         << " h_near_m: " << h_near_m << "m"
                // << std::endl;
    }
    if(have_far){
        got_height = true;
        yGround_far = percentile(gY_far, 0.1f);
        h_far_m = std::fabs(yFoot - yGround_far) * TO_M;
        // std::cout <<"[GROUND] yFoot: " << yFoot
        //             << " yGround_far: " << yGround_far
        //             << " h_far_m: " << h_far_m << "m"
        //             << std::endl;
    }

    size_t near_cnt = gY_near.size();
    size_t far_cnt = gY_far.size();                

    float chosen_h = -1.0f;
    if(have_near || have_far){
        bool both = have_near && have_far;
        if(both){
            const float SMALL_DIFF = 0.15f; // 15cm内平均
            if(h_near_m >= 0.0f && h_far_m >= 0.0f &&
                std::fabs(h_near_m - h_far_m) <= SMALL_DIFF) {
                    chosen_h = 0.5f * (h_near_m + h_far_m);
            }else{
                if(near_cnt > far_cnt) chosen_h = h_near_m;
                else if(far_cnt > near_cnt) chosen_h = h_far_m;
                else chosen_h = std::max(h_near_m, h_far_m);
            }       
        }else{
            chosen_h = have_near ? h_near_m : h_far_m;
        }
    }else{
        return false;
    }

    // 针对有时 脚面低于地面的bug
    float yGround_ref = NAN;
    if (have_near) yGround_ref = yGround_near;
    if(have_far && std::isfinite(yGround_far)){
        if(!std::isfinite(yGround_ref)) yGround_ref = yGround_far;
        else yGround_ref = 0.5f * (yGround_near + yGround_far);
    }
    if(std::isfinite(yFoot)&& std::isfinite(yGround_ref)&&(yFoot < yGround_ref - 100.0f)){
            return false;
    }
    
    height_m = chosen_h;

    if(got_height && height_m >= 0.0f){
        const float EPS_SMALL =0.08f;
        const float NEED_cnt =2;  

        if(stable_height < 0.0f){
            if (cand_height <0.0f || std::fabs(cand_height - height_m) > EPS_SMALL){
                cand_height = height_m;cand_count = 1;
            }else{
                cand_count++;
                if(cand_count >= NEED_cnt){
                    stable_height = cand_height;
                    cand_height = -1.0f;cand_count = 0;
                }
            }
        }else{
            float diff = std::abs(stable_height - height_m);
            if(diff < EPS_SMALL){
                cand_height = -1.0f;cand_count = 0;
            }else{
                if(cand_height < 0.0f || std::fabs(cand_height - height_m) > EPS_SMALL){
                    cand_height = height_m;cand_count = 1;
                }else {
                    cand_count++;
                    if(cand_count >= NEED_cnt){
                        stable_height = cand_height;
                        cand_height = -1.0f;cand_count = 0;
                    }
                }
            }
        }        
    }
    if(stable_height <0.0f) return false;
    out_height_m = stable_height;return true;
}

int main(int argc, char* argv[])
{
     //create
    auto pipeline = dai::Pipeline();
    auto camRgb     = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft   = pipeline.create<dai::node::ColorCamera>();
    auto monoRight  = pipeline.create<dai::node::ColorCamera>();
    auto depth      = pipeline.create<dai::node::StereoDepth>();
    auto pointCloud = pipeline.create<dai::node::PointCloud>();

    // 创建同步节点和输出节点
    auto sync = pipeline.create<dai::node::Sync>();
    auto xOut = pipeline.create<dai::node::XLinkOut>();
    xOut->input.setBlocking(false);

    //Properties
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    std::tuple<int,int> isp_scale_center(2,3);
    camRgb->setIspScale(isp_scale_center);
    camRgb->setFps(FPS);

    monoLeft->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    // monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    std::tuple<int,int> isp_scale_Left(2,3);
    monoLeft->setIspScale(isp_scale_Left);
    monoLeft->setFps(FPS);


    monoRight->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1200_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    // monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    std::tuple<int,int> isp_scale_Right(2,3);
    monoRight->setIspScale(isp_scale_Right);
    monoRight->setFps(FPS);

    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
    depth->setLeftRightCheck(true); //去掉反光假匹配
    depth->setSubpixel(true); //开启亚像素
    depth->initialConfig.setConfidenceThreshold(105);
    depth->setDepthAlign(dai::CameraBoardSocket::CAM_B);
    // int w = 1280;
    // int h = 800;
    int w = 640;
    int h = 400;
    depth->setOutputSize(w,h);
    // depth->setPreviewSize(1280, 800);  //w=1280，h=800
    // 

    //set
    pointCloud->initialConfig.setSparse(SPARSE);

    //link
    monoLeft->isp.link(depth->left);
    monoRight->isp.link(depth->right);
    depth->depth.link(pointCloud->inputDepth);
    camRgb->isp.link(sync->inputs["rgb"]);
    pointCloud->outputPointCloud.link(sync->inputs["pdata"]);
    sync->out.link(xOut->input);
    xOut->setStreamName("out");

    // device
    dai::Device device(pipeline);
    auto q = device.getOutputQueue("out", 4, false);

    // bool first = true;
    /* 初始化算法句柄 */

    //cv::Mat m_matImg = cv::imread("1.jpg");
    cv::Mat src=cv::Mat::zeros(cv::Size(w,h),CV_8UC3);


    CRbcahExample *example = new CRbcahExample(src);

    /* 配置算法 */
    example->config(src, false);
    RB_CAH_Result_S m_stResults;



    while(true) 
    {
        bool DEBUG_DRAW = true; //可视化开关
        auto inMessage = q->get<dai::MessageGroup>();//取出信息
        auto inColor = inMessage->get<dai::ImgFrame>("rgb");
        auto colorFrame = inColor->getCvFrame(); // 把depthai的图像帧转换为cv::mat

        float HEIGHT_THRE = 0.3f;

        //图像预处理
        cv::Mat resized_image;
        cv::resize(colorFrame, resized_image, cv::Size(320, 320));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);//BGR->RGB

        //获取点云
        auto inPointCloud=inMessage->get<dai::PointCloudData>("pdata");
        auto points= inPointCloud->getPoints();
        auto width=inPointCloud->getWidth();    
        auto height=inPointCloud->getHeight();
        // 复制点云数据到连续内存
        copyPointCloudToBuffer(points, width, height);
        const dai::Point3f* points_ptr = gPC.data.get();

        cv::Mat redst;
        cv::resize(colorFrame, redst, cv::Size(width, height));// 彩色图 colorFrame 按指定尺寸缩放到 redst

        /* 算法处理 */
        example->process(redst);
        std::cout<<"目标数量: "<<example->m_stResults.vecPedSet.size()<<std::endl;
        static int last_count = 0;
        //帧间去抖状态
        static float stable_height = -1.0f;
        static float cand_height = -1.0f;
        static int cand_count = 0;
        int now_count = example->m_stResults.vecPedSet.size();
        if (now_count == 0){
            stable_height = -1.0f;
            cand_height = -1.0f;
            cand_count = 0;
        }


        for (int i = 0; i < example->m_stResults.vecPedSet.size(); i++)
        {
            RB_CAH_Ped_S stTmp = example->m_stResults.vecPedSet[i];
            // 获取目标框的位置
            cv::Rect cvRect;
            cvRect.x = stTmp.stPedBoundingBox.stTopLeft.s32X;
            cvRect.y = stTmp.stPedBoundingBox.stTopLeft.s32Y;
            cvRect.width = stTmp.stPedBoundingBox.stBottomRight.s32X - cvRect.x;
            cvRect.height = stTmp.stPedBoundingBox.stBottomRight.s32Y - cvRect.y;
            cv::Point pt1, pt2, pt3, pt4;
            pt1.x = cvRect.x;
            pt1.y = cvRect.y;
            pt2.x = pt1.x + cvRect.width;
            pt2.y = pt1.y;
            pt3.x=pt2.x;
            pt3.y=pt2.y+cvRect.height;
            pt4.x=pt1.x;
            pt4.y=pt3.y;
            // 绘制 左上 右下 橙色
            cv::rectangle(redst,pt1,pt3,cv::Scalar(0, 128, 255), 2);
            std::string label = "Class:" + kClassNames[stTmp.s32labelID]+" "+std::to_string(stTmp.f32Prob);
            cv::putText(redst, label, cv::Point(pt1.x, pt1.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        // cv::imshow("color",redst);
        // 日志限频显示


        std::vector<PointCloud::Ptr> vcloud;
        std::vector<int> vlabel;
        std::vector<float> vprob;
        for (int target=0;target<example->m_stResults.vecPedSet.size();target++)
        {

            RB_CAH_Ped_S stTmp = example->m_stResults.vecPedSet[target];
            PointCloud::Ptr cloud (new PointCloud);
            vlabel.push_back(stTmp.s32labelID);
            vprob.push_back(stTmp.f32Prob);

            auto clampi = [](float v, float lo, float hi)->float {
                return std::max(lo, std::min(v, hi));
            };
            int u0 = stTmp.stPedBoundingBox.stTopLeft.s32X;
            int v0 = stTmp.stPedBoundingBox.stTopLeft.s32Y;
            int u1 = stTmp.stPedBoundingBox.stBottomRight.s32X;
            int v1 = stTmp.stPedBoundingBox.stBottomRight.s32Y;
            u0 = clampi(u0, 0, (int)width  - 1);
            u1 = clampi(u1, 0, (int)width  - 1);
            v0 = clampi(v0, 0, (int)height - 1);
            v1 = clampi(v1, 0, (int)height - 1);

            for (int y_Row = v0; y_Row < v1; y_Row++)
            {
                for (int x_Col = u0; x_Col < u1; x_Col++)
                {
                    int pose_ = y_Row * width + x_Col;
                    if(pose_ <0 || static_cast<size_t>(pose_) >= gPC.vaild) continue;
                    if(x_Col>u0 && x_Col<u1 && y_Row>v0 && y_Row<v1)
                    {
                        PointT p;
                        p.x = -points_ptr[pose_].x;
                        p.y = points_ptr[pose_].y;
                        p.z = points_ptr[pose_].z;
                        cv::Vec3b pixel = redst.at<cv::Vec3b>(y_Row, x_Col);
                        p.r = pixel[0];
                        p.g = pixel[1];
                        p.b = pixel[2];
                        cloud->points.push_back(p);
                    }
                }
            }
            cloud->height = 1;
            cloud->width = cloud->points.size();
            cloud->is_dense = false; // 允许后续滤波正确处理可能存在的无效点
            vcloud.push_back(cloud);
        }

        for (int target=0;target< example->m_stResults.vecPedSet.size();target++)
        {
            RB_CAH_Ped_S stTmp = example->m_stResults.vecPedSet[target];
            PointT min_point;
            PointT max_point;
            //排除噪声  
            Process_PointCloud(vcloud[target], min_point, max_point,50.0f,200,2500000);
            
            float out_height_m = -1.0f;
            bool ok = compute_height(gPC.data.get(),(int)width,(int)height,
                                    redst,stTmp,DEBUG_DRAW,
                                    stable_height,cand_height,cand_count,
                                    out_height_m);
            if(!ok){
                std::cout<<"本帧未得到稳定高度，或找不到可靠地面"<<std::endl;
                continue;
                }
            std::cout<<"距离地面高度:"<<out_height_m<<" m"<<std::endl;
            if(out_height_m > HEIGHT_THRE){
                std::cout<<"警告:目标高度超过阈值,开始检测是否佩戴安全绳和安全帽"<<std::endl;
                if (stTmp.s32labelID == 1 && stTmp.f32Prob >0.4f){
                    std::cout<<"检测到目标已佩戴安全绳但未佩戴安全帽!"<<std::endl;
                }else if(stTmp.s32labelID == 2 && stTmp.f32Prob >0.4f){
                    std::cout<<"检测到目标未佩戴安全绳且未佩戴安全帽!"<<std::endl;
                }else if(stTmp.s32labelID == 3 && stTmp.f32Prob >0.4f){
                    std::cout<<"检测到目标已佩戴安全帽但未佩戴安全绳!"<<std::endl;
                }
            }
        }    
    static auto lastShow = std::chrono::steady_clock::now();
    auto nowShow = std::chrono::steady_clock::now();
    float msSince = std::chrono::duration<float,std::milli>(nowShow - lastShow).count();
    if(msSince >= 33.3f){
        cv::imshow("color",redst);
        lastShow = nowShow;
    }
    last_count = now_count;
    // int key = cv::waitKey(1);
    int key = cv::waitKey(5);
    if(key == 'q' || key == 'Q') {
        return 0;
    }
    
    }
    /* 删除算法句柄 */
    delete example;


    return 0;
}

