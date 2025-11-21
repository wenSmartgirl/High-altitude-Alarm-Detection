
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
#include <pcl/filters/statistical_outlier_removal.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "rb_cah_detector.h"

using namespace std;
constexpr auto FPS = 30.0; 
bool SPARSE=false;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
const std::vector<std::string> kClassNames = {
    "No safety rope",
    "safety rope"//
};


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
    //wwdebug 距离补偿
    // if(dist > 0.0f){
    //     float factor = 1.0f;
    //     if(dist >=3.0f && dist <7.0f){
    //         factor = 1.0f-0.02f;
    //     }else if(dist >=7.0f && dist <12.0f){
    //         factor = 1.0f-0.03f;
    //     }else if(dist >=12.0f && dist <20.0f){
    //         factor = 1.0f-0.04f;
    //     }
    //     if(factor != 1.0f){
    //         for(auto& pt : cloud_down->points){
    //             pt.z *= factor;
    //         }
    //     }
    // }

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


    // 法线估计   需要足够多的邻居点
    if(cloud_down->size() >= 30){
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        tree->setInputCloud(cloud_down); // 当前点云构建成 KdTree
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setInputCloud(cloud_down); 
        ne.setSearchMethod(tree);// 使用 KdTree 查找邻居
        ne.setKSearch(16); // 使用最近的16个点来拟合平面
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        ne.compute(*normals);
    }

    //  在滤波之后计算 min/max
    pcl::getMinMax3D(*cloud_down, min_point, max_point);

    // 更新cloud
    cloud->swap(*cloud_down);
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


    depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
    depth->setLeftRightCheck(true); //去掉反光假匹配
    depth->setSubpixel(true); //开启亚像素
    depth->initialConfig.setConfidenceThreshold(220);
    depth->setDepthAlign(dai::CameraBoardSocket::CAM_B);
    int w = 1280;
    int h = 800;
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
            for (int y_Row = 0; y_Row < height; y_Row++)
            {
                for (int x_Col = 0; x_Col < width; x_Col++)
                {
                    // 二维坐标转换为一维数组
                    int pose_ = y_Row * width + x_Col;
                    
                    int u0,v0,u1,v1;
                    u0=stTmp.stPedBoundingBox.stTopLeft.s32X;
                    v0=stTmp.stPedBoundingBox.stTopLeft.s32Y;
                    u1=stTmp.stPedBoundingBox.stBottomRight.s32X;
                    v1=stTmp.stPedBoundingBox.stBottomRight.s32Y;
                    if(x_Col>u0 && x_Col<u1 && y_Row>v0 && y_Row<v1)
                    {
                        PointT p;
                        p.x = -points[pose_].x;
                        p.y = points[pose_].y;
                        p.z = points[pose_].z;
                        // 生成点云
                        cv::Vec3b pixel = redst.at<cv::Vec3b>(y_Row, x_Col);
                        // p.b = p.g =p.r = 255;
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
            // Process_PointCloud(vcloud[target], min_point, max_point);
            //排除噪声  
            Process_PointCloud(vcloud[target], min_point, max_point,50.0f,200,2500000);
            // L1范数 |dx|+|dy|+|dz|
            float val=abs(min_point.x-max_point.x)
                +abs(min_point.y-max_point.y)
                +abs(min_point.z-max_point.z);
           

            // ww优化 重写

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

            // DepthAI 点类型：dai::Point3f
            auto valid = [&](const dai::Point3f& p)->bool {
                // 有效深度范围
                const float Z_MIN = 10.0f;    // 0.1m
                const float Z_MAX = 30000.0f;  // 10 m
                return std::isfinite(p.y) && std::isfinite(p.z) && p.z >= Z_MIN && p.z <= Z_MAX;
            };

            // 取当前目标框并 clamp 到尺寸  bug1:width/height 是 unsigned  需要强行转换int
            int u0 = stTmp.stPedBoundingBox.stTopLeft.s32X;
            int v0 = stTmp.stPedBoundingBox.stTopLeft.s32Y;
            int u1 = stTmp.stPedBoundingBox.stBottomRight.s32X;
            int v1 = stTmp.stPedBoundingBox.stBottomRight.s32Y;
            u0 = clampi(u0, 0, static_cast<int>(width) - 1);
            u1 = clampi(u1, 0, static_cast<int>(width) - 1);
            v0 = clampi(v0, 0, static_cast<int>(height) - 1);
            v1 = clampi(v1, 0, static_cast<int>(height) - 1);

            // 脚带：底边上方3行×中间40%列
            int xL = u0 + (u1 - u0) * 2 / 10; // 左边30%
            int xR = u0 + (u1 - u0) * 8 / 10; // 右边30%
            xL = clampi(xL, u0, u1);
            xR = clampi(xR, u0, u1); //[30%-70%]
            // int footTop = std::max(v1 - 3, v0);
            // int footBot = std::max(v1 + 1, v0); // 底边
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
            // 遍历脚带位置提取所有有效点的Y和Z
            for(int yy = footTop; yy <= footBot; ++yy){
                int row = yy * static_cast<int>(width);
                for(int xx = xL; xx <= xR; ++xx){
                    const auto &p = points[row + xx];          // points 是 std::vector<dai::Point3f>
                    if(valid(p)) { footY.push_back(p.y); footZ.push_back(p.z); }
                }
            }

            std::cout << "脚带数量：" << footY.size() << std::endl;
            if(val > 0.01f && static_cast<int>(footY.size()) >= 2){
                float zFoot = median(footZ);   // 中位数深度
                //wwdebug
                float dist = zFoot * 0.001f; // 转为米
                float dist_m = std::max(0.3f,std::min(dist,20.0f));
                std::cout << "脚带中位数深度：" << dist << "米" << std::endl;
                //过滤Z
                float DZ_FOOT_MM = 150.0f; // 15cm
                if(dist_m >= 7.0f && dist_m <= 12.0f){
                    DZ_FOOT_MM = 200.0f; // 20cm
                }else if(dist_m > 12.0f){
                    DZ_FOOT_MM = 260.0f; 
                }
                
                std::vector<float> footY_filt; footY_filt.reserve(footY.size());
                for(size_t k = 0; k < footY.size(); ++k){
                    if(std::fabs(footZ[k] - zFoot) <= DZ_FOOT_MM){
                        footY_filt.push_back(footY[k]);
                    }
                }
                if(static_cast<int>(footY_filt.size()) < 2){
                    std::cout << "[WARN]脚点Z过滤后数量不足2个" << 
                        "footY_filt.size()=" << footY_filt.size() << 
                        "footY.size()=" << footY.size() << std::endl;
                    continue;
                }
                std::cout << "footY_filt.size()=" << footY_filt.size() << std::endl;

                float yFoot = percentile(footY_filt, 0.8f); // 
                float yFoot1 = percentile(footY_filt, 0.9f); //
                float yFoot2 = percentile(footY_filt, 0.5f); //
                std::cout << "yFoot=" << yFoot << std::endl;
                std::cout << "yFoot1=" << yFoot1 << std::endl;
                std::cout << "yFoot2=" << yFoot2 << std::endl;

                // 脚带 框底向下10行
                int gTop = std::min(v1+5,static_cast<int>(height) - 1);
                int gBot = std::min(v1+45, static_cast<int>(height) - 1);

                std::cout<<"zfoot="<<zFoot<<std::endl;
                // 近地面的点 和 脚的深度相差不超过
                // 近地面 120mm以内 远地面往后+40mm-300mm 上限900mm 或 深度的4%
                float Z_NEAR_MIN = 0.0f;
                float Z_NEAR_MAX = 120.0f;
                float DZ_FAR_MIN = 0.0f;
                float DZ_FAR_MAX = 0.0f;
                float FAR_EXTRA_MIN = 40.0f;
                float FAR_BAND = 300.0f;
                float FAR_CAP = 900.0f;
                float zFoot_ratio_cap = 0.04f;
                if(dist_m >= 0.0f && dist_m <= 7.0f){
                    Z_NEAR_MAX = 150.0f;
                    DZ_FAR_MIN = 350.0f;
                    DZ_FAR_MAX = 950.0f;
                }else if(dist_m >= 7.0f && dist_m <= 12.0f){
                    Z_NEAR_MAX = 160.0f;
                    gTop = std::min(v1+5,static_cast<int>(height) - 1);
                    gBot = std::min(v1+110, static_cast<int>(height) - 1);
                    DZ_FAR_MIN = std::max(Z_NEAR_MAX + 60.0f, 300.0f);
                    DZ_FAR_MAX = 1000.0f;
                }else if(dist_m > 12.0f){
                    Z_NEAR_MAX = 240.0f;
                    FAR_EXTRA_MIN = 80.0f;
                    FAR_BAND = 500.0f;
                    FAR_CAP = 1600.0f;
                    zFoot_ratio_cap = 0.06f;
                    gTop = std::min(v1+5,static_cast<int>(height) - 1);
                    gBot = std::min(v1+140, static_cast<int>(height) - 1);
                    DZ_FAR_MIN = std::max(Z_NEAR_MAX + FAR_EXTRA_MIN, 300.0f);
                    float temp1 = std::min(zFoot * zFoot_ratio_cap, FAR_CAP);
                    DZ_FAR_MAX = std::min(FAR_BAND + Z_NEAR_MAX,temp1);
                }

                std::cout << "DZ_FAR_MIN: " << DZ_FAR_MIN << " DZ_FAR_MAX: " << DZ_FAR_MAX << std::endl;
                std::cout << "Z_NEAR_MIN:" << Z_NEAR_MIN << " Z_NEAR_MAX: " << Z_NEAR_MAX << std::endl;
                if (DEBUG_DRAW) {
                    cv::rectangle(
                        redst,
                        cv::Rect(cv::Point(xL, gTop),
                                cv::Point(xR, gBot)),
                        cv::Scalar(0, 165, 255), 2  // 橙色
                    );
                    cv::putText(redst, "GROUND SEARCH",
                        cv::Point(xL, std::max(gTop - 5, 0)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,165,255), 1);
                }

                std::vector<float> gY_near; 
                gY_near.reserve((xR - xL + 1) * std::max(1, gBot - gTop + 1));
                std::vector<float> gY_far;
                gY_far.reserve((xR - xL + 1) * std::max(1, gBot - gTop + 1));
                for(int yy = gTop;yy <=gBot;++yy){
                    int row = yy * static_cast<int>(width);
                    for(int xx = xL;xx <= xR;++xx){
                        const auto &p = points[row + xx];
                        if(std::isnan(p.z)||p.z<10.0f||p.z>30000.0f) continue;
                        float dz = std::fabs(p.z - zFoot);
                        if(dz >= Z_NEAR_MIN && dz <= Z_NEAR_MAX){
                            gY_near.push_back(p.y);
                            if(DEBUG_DRAW)
                                redst.at<cv::Vec3b>(yy, xx) = cv::Vec3b(0,255,0);   // 近地面点：绿
                        }else if(dz >= DZ_FAR_MIN && dz <= DZ_FAR_MAX){
                            gY_far.push_back(p.y);
                            if(DEBUG_DRAW)
                                redst.at<cv::Vec3b>(yy, xx) = cv::Vec3b(255,0,0);   // 远地面点：蓝
                        }    
                    }
                }

                // if(msSince >= 33.3f){
                std::cout << "[DEBUG] footY size=" << footY.size() << std::endl;
                // }
                int MIN_CNT = 3;
                if(dist_m >= 7.0f && dist_m <= 12.0f){
                    MIN_CNT = 2;
                }else if (dist_m > 12.0f){
                    MIN_CNT = 1;
                }
                bool have_near = (static_cast<int>(gY_near.size()) >= MIN_CNT);
                bool have_far  = (static_cast<int>(gY_far.size())  >= MIN_CNT);
                std::cout << "[DEBUG] have_near=" << have_near
                          << " have_far=" << have_far << std::endl;
                std::cout << "[DEBUG] gY_near.size=" << gY_near.size()<<std::endl;
                std::cout << "[DEBUG] gY_far.size=" << gY_far.size()<<std::endl;

                static bool  floor_init = false; 

                float height_m = 0.0f;
                bool  got_height = false;
                constexpr float TO_M = 1.0f / 1000.0f;

                float h_near_m = -1.0f;
                float h_far_m = -1.0f;
                float yGround_near = NAN,yGround_far = NAN;
                if(have_near){
                    got_height = true;
                    yGround_near = percentile(gY_near, 0.1f);  
                    float yGround_near1 = percentile(gY_near, 0.5f);
                    float yGround_near2 = percentile(gY_near, 0.2f);
                    std::cout << "[DEBUG] yGround_near=" << yGround_near << std::endl;
                    std::cout << "[DEBUG] yGround_near1=" << yGround_near1 << std::endl;
                    std::cout << "[DEBUG] yGround_near2=" << yGround_near2 << std::endl;
                    h_near_m = std::fabs(yFoot - yGround_near) * TO_M; 
                    // if(msSince >= 33.3f){
                    std::cout <<"[GROUND] yFoot: " << yFoot
                            << " yGround_near: " << yGround_near
                            << " h_near_m: " << h_near_m << "m"
                            << std::endl;
                    // }
                    
                }
                if(have_far){
                    got_height = true;
                    yGround_far = percentile(gY_far, 0.6f);
                    float yGround_far1 = percentile(gY_far, 0.5f);
                    float yGround_far2 = percentile(gY_far, 0.2f);
                    std::cout << "[DEBUG] yGround_far=" << yGround_far << std::endl;
                    std::cout << "[DEBUG] yGround_far1=" << yGround_far1 << std::endl;
                    std::cout << "[DEBUG] yGround_far2=" << yGround_far2 << std::endl;
                    h_far_m = std::fabs(yFoot - yGround_far) * TO_M;
                    std::cout <<"[GROUND] yFoot: " << yFoot
                                << " yGround_far: " << yGround_far
                                << " h_far_m: " << h_far_m << "m"
                                << std::endl;
                }
                // // 看看走远地面还是走近地面
                size_t near_cnt = gY_near.size();
                size_t far_cnt = gY_far.size();
                std::string ground_scr = "NONE";
                
                const float LOW_HEIGHT = 0.15f;
                const float HIGH_HEIGHT = 0.25f;

                float chosen_h = -1.0f;
                if(have_near || have_far){
                    bool both = have_near && have_far;
                    if(both){
                       if(h_near_m >= HIGH_HEIGHT && h_far_m < LOW_HEIGHT){
                            chosen_h = h_near_m;
                            ground_scr = "NEAR";
                       }else if (h_near_m < LOW_HEIGHT && h_far_m >= HIGH_HEIGHT){
                           chosen_h = h_far_m;
                           ground_scr = "FAR";
                       }else{
                           chosen_h = 0.5f*(h_near_m + h_far_m);
                           ground_scr = "MID";
                       }
                            
                    }else{
                        if(have_near){
                            chosen_h = h_near_m;
                            ground_scr = "NEAR";
                        }else{
                            chosen_h = h_far_m;
                            ground_scr = "FAR";
                        }
                    }
                    got_height = true;
                }else{
                    ground_scr = "NONE";
                    got_height = false;
                }
                std::cout <<"[GROUND]near_cnt: " <<near_cnt
                            << " far_cnt: " << far_cnt
                            << " ground_scr: " << ground_scr
                            << " chosen_h: " << chosen_h << "m"
                            << " h_near_m: " << h_near_m << "m"
                            << " h_far_m: " << h_far_m << "m"
                            << std::endl;

                // 针对有时 脚面低于地面的bug
                float yGround_ref = NAN;
                if (have_near) yGround_ref = yGround_near;
                if(have_far && std::isfinite(yGround_far)){
                    if(!std::isfinite(yGround_ref)){
                        yGround_ref = yGround_far;
                    }else{
                        // 两个都有效取平均
                        yGround_ref = 0.5f * (yGround_near + yGround_far);
                    }
                }
                if(std::isfinite(yFoot)&& std::isfinite(yGround_ref)&&
                    (yFoot < yGround_ref - 100.0f)){
                        std::cout << "yFoot: "<<yFoot << " yGroud_ref: " << yGround_ref << std::endl;
                        std::cout << "异常：脚点高于地面点10cm,检测物体不稳定，跳过此帧" << std::endl;
                        continue;
                }
                
                height_m = chosen_h;

                if(got_height && height_m >= 0.0f){

                    const float EPS_SMALL =0.08f;
                    const float NEED_cnt =2;  

                    if(stable_height < 0.0f){
                        stable_height = height_m;
                        cand_height = -1.0f;
                        cand_count = 0;
                    }else{
                        float diff = std::abs(stable_height - height_m);
                        if(diff < EPS_SMALL){
                            cand_height = -1.0f;
                            cand_count = 0;
                        }else{
                            if(cand_height < 0.0f || std::fabs(cand_height - height_m) > EPS_SMALL){
                                cand_height = height_m;
                                cand_count = 1;
                            }else {
                                // 新高度和候选高度接近 计数+
                                cand_count++;
                                if(cand_count >= NEED_cnt){
                                    stable_height = cand_height;
                                    cand_height = -1.0f;
                                    cand_count = 0;
                                }
                            }
                        }
                    }
                    std::cout <<"稳定高度: "<< stable_height << "m,候选高度: "
                            << cand_height << "m,计数: " << cand_count << std::endl;

                    height_m = stable_height;
                    std::cout << "距离地面高度: " << height_m << "m" << std::endl;
                    if (height_m > HEIGHT_THRE) {
                        std::cout << "目标高度超过阈值,开始检测是否佩戴安全绳" << std::endl;
                        if (stTmp.s32labelID == 1 && stTmp.f32Prob > 0.4f) {
                            std::cout << "安全：目标已佩戴安全绳" << std::endl;
                        } else {
                            std::cout << "警告：该目标未佩戴安全绳！！" << std::endl;
                        }
                    }
                } else {
                    std::cout << "找不到可靠地面，请检查是否遮挡" << std::endl;
                }
            } else {
                std::cout << "脚带点太少或ROI过小" << std::endl;
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

