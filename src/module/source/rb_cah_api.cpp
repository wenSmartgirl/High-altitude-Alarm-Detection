
#include "rb_cah_api.h"
#include "rb_cah_gconfig.h"
#include "rb_cah_para.h"
#include "rb_cah_utils.h"

// 算法句柄结构体
typedef struct _rb_cah_handle_s
{
    CRbCAHUtils *__cahobj;
}RB_CAH_Handle_S;


/**
 * @brief 
 * 
 * @param  phCAH            算法句柄
 * @param  s32W             图像宽度
 * @param  s32H             图像高度
 * @return RB_EXPORTS 
 */
RB_STATUS_CODE_E RB_CAH_Create(RB_HANDLE* phCAH, RB_S32 s32W, RB_S32 s32H)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    // 创建cah实例
    RB_CAH_Handle_S* hCAH = new RB_CAH_Handle_S;
    if (NULL == hCAH)
    {
        return RB_FAILURE;
    }
    
    hCAH->__cahobj = new CRbCAHUtils(s32W, s32H);
    if (NULL == hCAH->__cahobj)
    {
        return RB_FAILURE;
    }
    
    // 返回句柄
    *phCAH = (RB_HANDLE)hCAH;

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAH_Destroy(RB_HANDLE hCAH)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAH_Handle_S* pHandle = (RB_CAH_Handle_S*)hCAH;
    if (NULL == pHandle)
    {
        return RB_FAILURE;
    }
    
    if (NULL == pHandle->__cahobj)
    {
        return RB_FAILURE;
    }
    else
    {
        delete pHandle->__cahobj;
        pHandle->__cahobj = NULL;
    }

    delete pHandle;
    pHandle = NULL;

    return s32Ret;
}


/**
 * @brief 
 * 
 * @param  pcLibVerStr      算法库版本号
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAH_GetLibVer(RB_S8* pcLibVerStr)
{
        sprintf((char*)pcLibVerStr, "\n=============== LIB INFO =====================\n"
            "Project: %s\n"
            "Version: %s\n"
            "Commit md5: %s\n"
            "Build DateTime: %s\n"
            "==============================================\n", 
            PROJECT_NAME,
            LIB_VERSION, 
            LIB_COMMIT_MD5,
            BUILD_DATETIME);

    return RB_SUCCESS;
}

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstCAHPara       CAH参数
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAH_Config(RB_HANDLE hCAH, const RB_CAH_Para_S* pstCAHPara)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAH_Handle_S* pHandle = (RB_CAH_Handle_S*)hCAH;
    if (NULL == pHandle || NULL == pstCAHPara)
    {
        return RB_FAILURE;
    }

    // 配置参数
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__cahobj->config(pstCAHPara));

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstImage         图像数据
 * @return 算法状态
 */
RB_STATUS_CODE_E RB_CAH_Process(RB_HANDLE hCAH, const RB_IMAGE_S* pstImage)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAH_Handle_S* pHandle = (RB_CAH_Handle_S*)hCAH;
    if (NULL == pHandle || NULL == pstImage || NULL == pstImage->pData)
    {
        return RB_FAILURE;
    }

#ifdef SHOW_DEBUG_TIME
    double dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    // 处理图像
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__cahobj->process(pstImage));
    if (s32Ret != RB_SUCCESS)
    {
        return s32Ret;
    }

#ifdef SHOW_DEBUG_TIME
    double dtEnd = __get_current_time_proc();
    std::cout<<"++++++  In CAH API: Process Time Span ++++++ "<< (dtEnd - dtStart) <<" ms"<<std::endl;
    dtStart = __get_current_time_proc();
#endif // SHOW_DEBUG_TIME

    return s32Ret;
}

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstCAHResult     CAH结果
 * @return 算法状态 
 */
RB_STATUS_CODE_E RB_CAH_GetResults(RB_HANDLE hCAH, RB_CAH_Result_S* pstCAHResult)
{
    RB_STATUS_CODE_E s32Ret = RB_SUCCESS;
    RB_CAH_Handle_S* pHandle = (RB_CAH_Handle_S*)hCAH;
    if (NULL == pHandle || NULL == pstCAHResult)
    {
        return RB_FAILURE;
    }

    // 获取结果，通过参数返回出来
    s32Ret = (RB_STATUS_CODE_E)(pHandle->__cahobj->get_results(pstCAHResult));

    return s32Ret;
}
