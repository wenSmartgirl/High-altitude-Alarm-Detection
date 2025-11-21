
#ifndef __RB_DETECTOR_H__
#define __RB_DETECTOR_H__

#include "rb_cah_gconfig.h"

/* 目标结构体 */
typedef struct __rb_det_target_s
{
    int s32labelID;        // 目标标签的 ID
    float f32Prob;         // 目标概率
    cv::Rect_<float> rect; // 目标包围框
} RB_DetTarget_S;


#ifdef WITH_NCNN  /* 使用 NCNN 作为前向推理框架 */

#include "layer.h"
#include "net.h"

#ifdef LOAD_MEM_MODEL
#include "rb_cah_id.h"
#include "rb_cah_mem.h"
#endif


/**
 * @brief 基于 ncnn的yolov5 的前向推理目标检测框架
 * 可以通过模型文件加载初始化模型，也可以通过内存的方式加载模型文件
 * 
 */
class CYOLOV5_NCNN_Detector
{
public:
    /* 初始化模型和各种参数 */
#ifdef LOAD_MEM_MODEL
    /* 通过内存初始化模型 */
    CYOLOV5_NCNN_Detector(const unsigned char *paramem, const unsigned char *modelmem);
#else
    /* 通过文件初始化模型 */
    CYOLOV5_NCNN_Detector(const char *parampath, const char *modelpath);
#endif

    ~CYOLOV5_NCNN_Detector();

    // 处理图像，输入的是 bgr 图像
    int process(const cv::Mat matIm, std::vector<RB_DetTarget_S> & objects);

    // 获得结果
    void get_results(std::vector<RB_DetTarget_S> & objects);

#ifdef SHOW_DEBUG_IMG
    // 显示图像
    void draw_objects();
#endif

    
private:
    // 内部常用内联函数
#ifdef LOAD_MEM_MODEL
    // 通过内存加载模型
    int __load_para_model_mem(const unsigned char* paramen, const unsigned char* modelmem);
#else
    // 通过路径加载模型
    int __load_para_model_file(const char *parampath, const char *modelpath);
#endif

    // 初始化网络模型的参数
    void __init_para();

    // 相交面积
    float __intersection_area(const RB_DetTarget_S &a, const RB_DetTarget_S &b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    void __qsort_descent(std::vector<RB_DetTarget_S> &vecObjs)
    {
        if (vecObjs.empty())
            return;

        __qsort_descent_inplace(vecObjs, 0, vecObjs.size() - 1);
    }

    // 将图像转换为 ncnn mat
    int __pre_process(const cv::Mat matIm);

    // NMS 合并目标框
    void __nms_sorted_bboxes(const std::vector<RB_DetTarget_S> &vecObjs, 
        std::vector<int> &picked);

    // 快速排序，将目标按照分数排序
    void __qsort_descent_inplace(std::vector<RB_DetTarget_S> &vecObjs, int left, int right);

    // 生成 proposal
    void __generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
        const ncnn::Mat &feat_blob, float prob_threshold, std::vector<RB_DetTarget_S> &objects);

    // 后处理
    void __post_process();

private:
    int m_s32ImgW;
    int m_s32ImgH;
    int m_s32PadW;
    int m_s32PadH;

    float m_f32Scale;

    int m_s32TgtSize;   // targe image max size
    float m_f32ProbThr; // prob threshold
    float m_f32NMSThr;  // NMS threshold
    float m_af32Norms[3];

    int m_s32Stride8;
    int m_s32Stride16;
    int m_s32Stride32;

    ncnn::Net m_netYOLOv5; // network

    ncnn::Mat m_matAnchors8;  // anchor @ stride 8
    ncnn::Mat m_matAnchors16; // anchor @ stride 16
    ncnn::Mat m_matAnchors32; // anchor @ stride 32
    ncnn::Mat in_pad;    

    // 输入的 bgr图像
    cv::Mat m_matBGRIn;

    std::vector<RB_DetTarget_S> m_vecProposals;
    std::vector<RB_DetTarget_S> m_vecObjs;
};


#else

#ifdef WITH_RKNN

#include <dlfcn.h>
#include <math.h>
#include <set>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "im2d_type.h"
#include "rga.h"
#include "rknn_api.h"

/**
 * @brief 目标检测类
 * 目标检测模型为 YOLOv5
 * 基于 RK3588 的 NPU 进行开发
 *
 */
class CYOLOV5_NPU_Detector_CAH
{
public:
    // 初始化模型和各种参数
    CYOLOV5_NPU_Detector_CAH();

    // 释放相关资源
    ~CYOLOV5_NPU_Detector_CAH();

    // 处理图像，输入的是 bgr 图像
    int process2(const cv::Mat matIm, std::vector<RB_DetTarget_S> &objects);

    // 获得结果
    int get_results(std::vector<RB_DetTarget_S> &objects);

#ifdef SHOW_DEBUG_IMG
    // 画目标
    void draw_objects();
#endif

private:
#ifdef SHOW_DEBUG_INFO
    // 打印相关信息
    void __dump_tensor_attr(rknn_tensor_attr *attr);
#endif

    // 加载模型
    unsigned char *__load_model(const char *filename, int *model_size);

    // 加载数据
    unsigned char *__load_data(FILE *fp, size_t ofst, size_t sz);

    // 加载 rknn 模型
    int __init_model();

    // NMS 合并目标框
    void __nms_sorted_bboxes(const std::vector<RB_DetTarget_S> &vecObjs, std::vector<int> &picked);

    // 快速排序，将目标按照分数排序
    void __qsort_descent_inplace(std::vector<RB_DetTarget_S> &vecObjs, int left, int right);

    // 生成 proposal
    void __generate_proposals(int8_t *input, int *anchor, int stride, int32_t zp,
                              float scale, std::vector<RB_DetTarget_S> &vecObjs);

    // 图像预处理
    // flag = 1 表示使用 rga 进行处理
    int __pre_process_rga(const cv::Mat matIm);

    // 预处理，将图像处理成
    int __pre_process(const cv::Mat matIm);

    // 后处理
    void __post_process(int8_t *input0, int8_t *input1, int8_t *input2);

private:
    // 输入的 bgr图像
    cv::Mat m_matBGRIn;

    // 前处理后的图像
    cv::Mat m_matRGBIn;

    // 图像宽高通道
    int m_ImgW;
    int m_ImgH;
    int m_ImgC;

    // 网络宽高和通道
    int m_NetW;
    int m_NetH;
    int m_NetC;

    // 图像与网络对应缩放比例
    float m_scaleW;
    float m_scaleH;
    int m_topPadd; //
    int m_bottomPadd;
    int m_leftPadd;
    int m_rightPadd;

    // 步长
    int m_stride8;
    int m_stride16;
    int m_stride32;

    // 锚框
    int m_anchor8[6];  // anchor @ stride 8
    int m_anchor16[6]; // anchor @ stride 16
    int m_anchor32[6]; // anchor @ stride 32

    // 检测出来的目标集合
    std::vector<RB_DetTarget_S> m_vecObjs;

    // rga 数据 和 rknn 相关数据
    rknn_context ctx;
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    int model_data_size;
    unsigned char *model_data;
    rknn_input inputs[1];
    void *resize_buf;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

    // 内部常用内联函数
private:
    // 初始化步长和锚框
    void __init_stride_anchor()
    {
        // 步长
        m_stride8 = 8;
        m_stride16 = 16;
        m_stride32 = 32;

        // anchors
        m_anchor8[0] = 10;
        m_anchor8[1] = 13;
        m_anchor8[2] = 16;
        m_anchor8[3] = 30;
        m_anchor8[4] = 33;
        m_anchor8[5] = 23;

        m_anchor16[0] = 30;
        m_anchor16[1] = 61;
        m_anchor16[2] = 62;
        m_anchor16[3] = 45;
        m_anchor16[4] = 59;
        m_anchor16[5] = 119;

        m_anchor32[0] = 116;
        m_anchor32[1] = 90;
        m_anchor32[2] = 156;
        m_anchor32[3] = 198;
        m_anchor32[4] = 373;
        m_anchor32[5] = 326;

        // // anchors
        // m_anchor8[0] = 175;
        // m_anchor8[1] = 88;
        // m_anchor8[2] = 192;
        // m_anchor8[3] = 179;
        // m_anchor8[4] = 314;
        // m_anchor8[5] = 141;

        // m_anchor16[0] = 267;
        // m_anchor16[1] = 231;
        // m_anchor16[2] = 201;
        // m_anchor16[3] = 334;
        // m_anchor16[4] = 320;
        // m_anchor16[5] = 328;

        // m_anchor32[0] = 419;
        // m_anchor32[1] = 300;
        // m_anchor32[2] = 419;
        // m_anchor32[3] = 371;
        // m_anchor32[4] = 512;
        // m_anchor32[5] = 355;



        m_scaleH = 1.f;
        m_scaleW = 1.f;
        m_topPadd = 0;
        m_bottomPadd = 0;
        m_leftPadd = 0;
        m_rightPadd = 0;
    }

    // 截断
    int __clamp(float val, int min, int max)
    {
        return val > min ? (val < max ? val : max) : min;
    }

    // 截断数据
    int32_t __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    // 相交面积
    float __intersection_area(const RB_DetTarget_S &a, const RB_DetTarget_S &b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    // __sigmoid
    float __sigmoid(float x)
    {
        return 1.0 / (1.0 + expf(-x));
    }

    // __unsigmoid
    float __unsigmoid(float y)
    {
        return -1.0 * logf((1.0 / y) - 1.0);
    }

    // 定点化
    int8_t __qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)__clip(dst_val, -128, 127);
        return res;
    }

    // 浮点化
    float __deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    void __qsort_descent(std::vector<RB_DetTarget_S> &vecObjs)
    {
        if (vecObjs.empty())
            return;

        __qsort_descent_inplace(vecObjs, 0, vecObjs.size() - 1);
    }
};
#endif

#endif

#endif // !__RB_DETECTOR_H__
