
#include "rb_cah_utils.h"

/* 构造函数 */
CRbCAHUtils::CRbCAHUtils(RB_S32 s32W, RB_S32 s32H)
{
    /* 清理目标信息 */
    __clean_objs();

    /* 初始化参数 */
    __init_para();

    /* 初始化模型，加载模型参数 */
    __init_model();

    /* 设置尺寸信息*/
    __set_size(s32W, s32H);
}

// 删除目标检测器
CRbCAHUtils::~CRbCAHUtils()
{
    /* 清理目标信息 */
    __clean_objs();
    delete m_PCYOLOV5DetNet;
    m_PCYOLOV5DetNet = NULL;
}

/* 清理目标信息 */
void CRbCAHUtils::__clean_objs()
{
    m_vecObjs.clear();
    vector<RB_DetTarget_S>().swap(m_vecObjs);
}

/* 初始化参数 */
void CRbCAHUtils::__init_para()
{
    m_s32ImW = 0;
    m_s32ImH = 0;
    m_s32BufLen = BUFFER_LEN;
    m_s32ROINum = DEFAULT_ROI_NUM;
    m_dOverlapRatio = OVERLAP_THR;
    m_s32ScaleW = 0;
    m_s32ScaleH = 0;
    m_f32MaskScale = 0.5;
    m_enDangerSens = RB_CAH_DANGER_SENS_DEFAULT;
}

/* 初始化模型，加载模型参数 */
void CRbCAHUtils::__init_model()
{
#ifdef WITH_NCNN
#ifdef LOAD_MEM_MODEL
    m_PCYOLOV5DetNet = new CYOLOV5_NCNN_Detector(NCNN_PARAM_MEM, NCNN_BIN_MEM);
#else
    m_PCYOLOV5DetNet = new CYOLOV5_NCNN_Detector(NCNN_PARAM_PATH, NCNN_BIN_PATH);
#endif
#else
#ifdef WITH_RKNN
    /* 初始化网络 */
    m_PCYOLOV5DetNet = new CYOLOV5_NPU_Detector_CAH;
#endif
#endif
}

/* 设置尺寸以及利用尺寸初始化 */
void CRbCAHUtils::__set_size(RB_S32 s32W, RB_S32 s32H)
{
    m_s32ImW = s32W;
    m_s32ImH = s32H;

    // 初始化opencv的BGR图像
    m_bgrIn = cv::Mat(s32H, s32W, CV_8UC3);

    // 默认为4个点的ROI
    RB_POINT_S astROIPoints[DEFAULT_ROI_VER_NUM];
    astROIPoints[0].s32X = INNER_BORDER;
    astROIPoints[0].s32Y = INNER_BORDER;
    astROIPoints[1].s32X = INNER_BORDER;
    astROIPoints[1].s32Y = s32H - INNER_BORDER;
    astROIPoints[2].s32X = s32W - INNER_BORDER;
    astROIPoints[2].s32Y = s32H - INNER_BORDER;
    astROIPoints[3].s32X = s32W - INNER_BORDER;
    astROIPoints[3].s32Y = INNER_BORDER;
    for (int i = 0; i < DEFAULT_ROI_VER_NUM; i++)
    {
        m_avecROIPolyGon[0].push_back(astROIPoints[i]);
    }

    m_f32MaskScale = 240.f / (float)s32W;
    m_s32ScaleW = m_s32ImW * m_f32MaskScale;
    m_s32ScaleH = m_s32ImH * m_f32MaskScale;
    m_matROIMask = cv::Mat(m_s32ScaleH, m_s32ScaleW, CV_8UC1, cv::Scalar(255));
}

int CRbCAHUtils::config(const RB_CAH_Para_S *pstPara)
{
    if (NULL == pstPara)
    {
        return RB_FAILURE;
    }

    for (int i = 0; i < MAX_CAH_ROI; i++)
    {
        vector<RB_POINT_S>().swap(m_avecROIPolyGon[i]);
    }
    

    switch (m_enDangerSens)
    {
    case RB_CAH_DANGER_SENS_LOW:
        m_dOverlapRatio *= LOW_SENS_RATIO;
        break;
    case RB_CAH_DANGER_SENS_HIGH:
        m_dOverlapRatio *= HIGH_SENES_RATIO;
        break;
    default:
        m_dOverlapRatio *= 1;
        break;
    }

    std::vector< std::vector<cv::Point> > vvPts;
    std::vector<cv::Point> vPts;
    cv::Point ptTmp;

    for (int i = 0; i < m_s32ROINum; i++)
    {
        RB_POINT_S stPTTmp;
        int ptNum = pstPara->vecROIPts[i].size();
        if (ptNum > MAX_VER_NUM)
        {
            ptNum = MAX_VER_NUM;
        }

        for (int j = 0; j < ptNum; j++)
        {
            stPTTmp = pstPara->vecROIPts[i][j];
            m_avecROIPolyGon[i].push_back(stPTTmp);

            ptTmp.x = (int)(stPTTmp.s32X * m_f32MaskScale);
            ptTmp.y = (int)(stPTTmp.s32Y * m_f32MaskScale);
            vPts.push_back(ptTmp);
        }

        vvPts.push_back(vPts);
        std::vector<cv::Point>().swap(vPts);
    }

    m_matROIMask = cv::Mat::zeros(m_s32ScaleH, m_s32ScaleW, CV_8UC1);


    cv::fillPoly(m_matROIMask, vvPts, cv::Scalar(255, 255, 255), 8, 0);
    std::vector< std::vector<cv::Point> >().swap(vvPts);

    // cv::imshow("ROIMask", m_matROIMask);
    // cv::waitKey(0);
    
    return RB_SUCCESS;
}


// DONE
int CRbCAHUtils::process(const RB_IMAGE_S* pstImage)
{
    int ret;

#ifdef SHOW_DEBUG_TIME
    double dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    // 图像转换
    m_bgrIn = convert2CvMat(pstImage);

    // int hh=m_bgrIn.rows;
    // int ww=m_bgrIn.cols;
    // std::cout<<"Image Height: "<<hh<<std::endl;
    // std::cout<<"Image Width:  "<<ww<<std::endl;
    // cv::imwrite("save.jpg",m_bgrIn);


#ifdef SHOW_DEBUG_TIME
    double dtEnd = __get_current_time_proc();
    std::cout<<"++++++  In CRbCASUtils: Convert Image Time Span ++++++ "<< (dtEnd - dtStart) <<" ms"<<std::endl;
    dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    // 检测
    m_vecObjs.clear();
    vector<RB_DetTarget_S>().swap(m_vecObjs);
    ret = m_PCYOLOV5DetNet->process2(m_bgrIn, m_vecObjs);
    if (ret != 0)
    {
        return ret;
    }


#ifdef SHOW_DEBUG_IMG
    m_PCYOLOV5DetNet->draw_objects();
#endif

#ifdef SHOW_DEBUG_TIME
    dtEnd = __get_current_time_proc();
    std::cout<<"++++++  In CRbCAHUtils: Network Process Image Time Span ++++++ "<< (dtEnd - dtStart) <<" ms"<<std::endl;
#endif // SHOW_DEBUG_TIME

    // 清空数据
    std::vector<RB_CAH_Ped_S>().swap(m_stResults.vecPedSet);

    std::vector<cv::Mat>().swap(m_vecMatTgtMask);

    RB_RECT_S stBB;
    for (size_t i = 0; i < m_vecObjs.size(); i++)
    {
        const RB_DetTarget_S & obj = m_vecObjs[i];
        RB_CAH_Ped_S stPed;
        stBB.stTopLeft.s32X = obj.rect.x;
        stBB.stTopLeft.s32Y = obj.rect.y;
        stBB.stBottomRight.s32X = obj.rect.width + obj.rect.x;
        stBB.stBottomRight.s32Y = obj.rect.height + obj.rect.y;


        stPed.f32Prob = obj.f32Prob;
        stPed.s32labelID=obj.s32labelID;
        stPed.stPedBoundingBox = stBB;
       
        m_stResults.vecPedSet.push_back(stPed);
    }

    __post_process();

    return ret;
}

// 规则后处理
int CRbCAHUtils::__post_process()
{
    // TODO: 
  
    //



    
    
    
    return RB_SUCCESS;
}

// 获得结果
int CRbCAHUtils::get_results(RB_CAH_Result_S* pstResult)
{
    if (NULL == pstResult)
    {
        return RB_FAILURE;
    }

 
    std::vector<RB_CAH_Ped_S>().swap(pstResult->vecPedSet);

    for (int i = 0; i < m_stResults.vecPedSet.size(); i++)
    {
        RB_CAH_Ped_S stCAHPed = m_stResults.vecPedSet[i];
        pstResult->vecPedSet.push_back(stCAHPed);
    }

   
    return RB_SUCCESS;
}
