
#ifndef __RB_CAH_API_H__
#define __RB_CAH_API_H__

#include "rb_common.h"
#include "rb_cah_para.h"


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief 
 * 
 * @param  phCAH            算法句柄
 * @param  s32W             图像宽度
 * @param  s32H             图像高度
 * @return RB_EXPORTS 
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_Create(RB_HANDLE* phCAH, RB_S32 s32W, RB_S32 s32H);

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_Destroy(RB_HANDLE hCAH);


/**
 * @brief 
 * 
 * @param  pcLibVerStr      算法库版本号
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_GetLibVer(RB_S8* pcLibVerStr);

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstCAHPara       CAH参数
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_Config(RB_HANDLE hCAH, const RB_CAH_Para_S* pstCAHPara);

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstImage         图像数据
 * @return 算法状态
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_Process(RB_HANDLE hCAH, const RB_IMAGE_S* pstImage);

/**
 * @brief 
 * 
 * @param  hCAH             算法句柄
 * @param  pstCAHResult     CAH结果
 * @return 算法状态 
 */
RB_EXPORTS RB_STATUS_CODE_E RB_CAH_GetResults(RB_HANDLE hCAH, RB_CAH_Result_S* pstCAHResult);


#ifdef __cplusplus
}
#endif

#endif // !__RB_CAH_API_H__
