

#ifndef __CAH_EXAMPLE_UTILS_H__
#define __CAH_EXAMPLE_UTILS_H__

#include "rb_common.h"
#include "rb_cah_api.h"
#include "rb_cah_gconfig.h"
#include "rb_cah_para.h"

/* 算法用例处理的类 */
class CRbcahExample
{
public:
    /* 通过图像宽高创建对象 */
    CRbcahExample(const int w, const int h);

    /* 通过图像创建对象 */
    CRbcahExample(cv::Mat matImgInit);

    /* 释放资源 */
    ~CRbcahExample();

    /* 配置ROI信息，其他信息默认 */
    int config(cv::Mat matImgConfig, bool bManual = false );

    /* 用例处理 */
    int process();

    /* 处理每一帧数据 */
    int process(cv::Mat matImgProcess);

    /* 显示结果 */
    int show_results(int wait_time=0, float show_ratio=1.f);

public: 
    /* 图像操作对象 */
    cv::Mat m_matResult;

    /* 内部结果 */
    RB_CAH_Result_S m_stResults;

private:
    /* 初始化数据 */
    int __init(int w, int h);

    /* 鼠标选择ROI区域 */
    cv::Rect mouse_selectROI(cv::Mat matImgConfig);

    /* 鼠标选择点集 */
    std::vector<cv::Point> mouse_selecPtsSet(cv::Mat matImgConfig);





private:
    /* 图像宽高 */
    int m_iImgWidth;
    int m_iImgHeight;

    /* ROI */
    cv::Rect m_rcROI;

    /* 算法处理的图像 */
    cv::Mat m_matImg;

    /* 区域点集 */
    std::vector<cv::Point> m_vPoints;

    /* 算法句柄 */
    RB_HANDLE m_AlgoHandle;

    /* 内部参数 */
    RB_CAH_Para_S m_stPara;


};

#endif 
