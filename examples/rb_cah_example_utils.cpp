
#include "rb_cah_example_utils.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define INNER_BORDER 5

CRbcahExample::CRbcahExample(const int w, const int h)
{
    m_iImgHeight = h;
    m_iImgWidth = w;

    /* 创建一个空的图像数据 */
    m_matImg = cv::Mat::zeros(w, h, CV_8UC3);

    /* 初始化 */
    __init(w, h);
}

CRbcahExample::CRbcahExample(cv::Mat matImgInit)
{
    /* 判断图像是否为空 */
    if (matImgInit.empty())
    {
        std::cout << "Image is empty, please confirm the input!" << std::endl;
    }

    /* 拷贝图像 */
    matImgInit.copyTo(m_matImg);

    m_iImgWidth = m_matImg.cols;
    m_iImgHeight = m_matImg.rows;

    __init(m_iImgWidth, m_iImgHeight);
}

/* 内部初始化 */
int CRbcahExample::__init(int w, int h)
{
    int ret = RB_SUCCESS;

    /* 创建算法句柄 */
    ret = RB_CAH_Create(&m_AlgoHandle, w, h);

    /* 打印算法版本号 */
    char version[256];
    ret = RB_CAH_GetLibVer(version);
    std::cout << version << std::endl;

    return ret;
}

/* 释放用例和算法句柄 */
CRbcahExample::~CRbcahExample()
{
    /* 释放算法句柄 */
    RB_CAH_Destroy(m_AlgoHandle);
}

/* 通过鼠标选择 ROI */
cv::Rect CRbcahExample::mouse_selectROI(cv::Mat matImgConfig)
{
    /* 默认 ROI 为全屏 */
    cv::Rect roi(0, 0, 0, 0);
    cv::Mat frame = matImgConfig.clone();
    cv::Point anchor;
    bool working = false;

#define WINDOW_NAME "Config"
#define ROI_WINDOW "ROI"

    cvui::init(WINDOW_NAME);

    while (true)
    {
        /* 拷贝图像 */
        matImgConfig.copyTo(frame);

        /*打印提示信息 */
        cvui::text(frame, 10, 10, "Click (any) mouse button and drag the pointer around to select a ROI.", 0.5);

        /*按下鼠标 */
        if (cvui::mouse(cvui::DOWN))
        {
            anchor.x = cvui::mouse().x;
            anchor.y = cvui::mouse().y;
            working = true;
        }

        /* 确定是否按下鼠标 */
        if (cvui::mouse(cvui::IS_DOWN))
        {
            int width = cvui::mouse().x - anchor.x;
            int height = cvui::mouse().y - anchor.y;

            roi.x = width < 0 ? anchor.x + width : anchor.x;
            roi.y = height < 0 ? anchor.y + height : anchor.y;
            roi.width = std::abs(width);
            roi.height = std::abs(height);

            cvui::printf(frame, roi.x + 5, roi.y + 5,
                         0.5, 0xee17dd, "(%d, %d)",
                         roi.x, roi.y);
            cvui::printf(frame, cvui::mouse().x + 5, cvui::mouse().y + 5,
                         0.5, 0xee17dd, "%d x %d",
                         roi.width, roi.height);
        }

        /* 鼠标弹起 */
        if (cvui::mouse(cvui::UP))
        {
            working = false;
        }

        /* 确保 ROI 的边界 */
        roi.x = roi.x < 0 ? 0 : roi.x;
        roi.y = roi.y < 0 ? 0 : roi.y;
        roi.width = roi.x + roi.width > frame.cols - 1 ? roi.width + frame.cols - (roi.x + roi.width) : roi.width;
        roi.height = roi.y + roi.height > frame.rows - 1 ? roi.height + frame.rows - (roi.y + roi.height) : roi.height;

        /* 绘制 ROI */
        cvui::rect(frame, roi.x, roi.y, roi.width,
                   roi.height, 0x46ee17);

        /* 更新 UI */
        cvui::update();

        /* 显示图像 */
        cv::imshow(WINDOW_NAME, frame);

        /* ROI 有效的话就显示 */
        if (roi.area() > 0 && !working)
        {
            cv::imshow(ROI_WINDOW, matImgConfig(roi));
        }

        if (cv::waitKey(20) == 's' || cv::waitKey(20) == 'S')
        {
            break;
        }
    }

    cv::destroyAllWindows();
#undef WINDOW_NAME
#undef ROI_WINDOW

    return roi;
}

/* 通过鼠标选择 ROI */
std::vector<cv::Point> CRbcahExample::mouse_selecPtsSet(cv::Mat matImgConfig)
{
    std::vector<cv::Point> pts;
    // /* 默认 ROI 为全屏 */
    cv::Rect roi(0, 0, 0, 0);
    cv::Mat frame = matImgConfig.clone();
    cv::Point anchor;
    bool working = false;

#define WINDOW_NAME "Config2"
#define ROI_WINDOW "ROI2"

    cvui::init(WINDOW_NAME);

    /* 点的个数 */
    int pt_num = 0;
    int w, h;
    cv::Point pt1, pt2, pt3, pt4;
    w = matImgConfig.cols;
    h = matImgConfig.rows;

    while (true)
    {
        /* 拷贝图像 */
        matImgConfig.copyTo(frame);

        /*打印提示信息 */
        cvui::text(frame, 10, 10, "Click (any) mouse button select Points, press S to save, press R to redraw.", 0.5);

        /* 按下一个点，插入记录一个点 */
        if (cvui::mouse(cvui::DOWN))
        {
            pt_num += 1;
            anchor.x = cvui::mouse().x;
            anchor.y = cvui::mouse().y;

            // /* 第三个点的纵坐标和第二点一样 */
            // if (pt_num == 3)
            // {
            //     anchor.y = pts[1].y;
            // }

            // /* 第四个点的纵坐标和第一点一样 */
            // if (pt_num == 4)
            // {
            //     anchor.y = pts[0].y;
            // }

            pts.push_back(anchor);
        }

        /* 画十字线 */
        {
            anchor.x = cvui::mouse().x;
            anchor.y = cvui::mouse().y;
            pt1 = anchor, pt2 = anchor, pt3 = anchor, pt4 = anchor;
            pt1.y = 0, pt2.y = h - 1, pt3.x = 0, pt4.x = w - 1;

            cv::line(frame, pt1, pt2, cv::Scalar(0, 255, 0), 1, 8, 0);
            cv::line(frame, pt3, pt4, cv::Scalar(0, 255, 0), 1, 8, 0);
        }

        if (pt_num > 0)
        {
            int show_num = pt_num;
            if (pt_num > 4)
            {
                std::cout << "Select more than 4 points" << std::endl;
                cvui::text(frame, 10, 30, "Select more than 4 points, press S to save, press R to redraw.", 0.5);
                show_num = 4;
            }

            for (int i = 0; i < show_num; i++)
            {
                cv::circle(frame, pts[i], 5, cv::Scalar(0, 0, 255), -1);
                cv::line(frame, pts[i], pts[(i + 1) % show_num], cv::Scalar(0, 255, 255), 2);

                pt1 = pts[i], pt2 = pts[i], pt3 = pts[i], pt4 = pts[i];
                pt1.y = 0, pt2.y = h - 1, pt3.x = 0, pt4.x = w - 1;

                cv::line(frame, pt1, pt2, cv::Scalar(0, 255, 0), 1, 8, 0);
                cv::line(frame, pt3, pt4, cv::Scalar(0, 255, 0), 1, 8, 0);
            }

            if (pt_num < 4)
            {
                anchor.x = cvui::mouse().x;
                anchor.y = cvui::mouse().y;
                cv::line(frame, pts[pt_num - 1], anchor, cv::Scalar(0, 128, 255), 2, 8, 0);
            }
        }

        /* 更新 UI */
        cvui::update();

        /* 显示图像 */
        cv::imshow(WINDOW_NAME, frame);

        char key = cv::waitKey(20);

        if (key == 's' || key == 'S')
        {
            if (pts.size() >= 4)
            {
                break;
            }
            else
            {
                std::cout << "Can't save the selected points, because the pt num is not enough!" << std::endl;
            }
        }
        else if (key == 'r' || key == 'R')
        {
            /* 清空数据，重新绘制 */
            pts.clear();
            std::vector<cv::Point>().swap(pts);
            pt_num = 0;
            std::cout << "Select more than 4 points" << std::endl;
        }
    }

    cv::destroyAllWindows();
#undef WINDOW_NAME
#undef ROI_WINDOW

    return pts;
}

/* 通过该 config 函数配置 ROI信息, 其他信息默认 */
int CRbcahExample::config(cv::Mat matImgConfig, bool bManual)
{

    int ret = RB_SUCCESS;
    RB_CAH_Para_S stPara;
    int w, h;
    w = matImgConfig.cols;
    h = matImgConfig.rows;
    stPara.enSens = RB_CAH_DANGER_SENS_DEFAULT;
    stPara.s32BufferLen = 5;
    stPara.s32ROINum = 1;

    /* 手动还是自动 */
    if (bManual) /* 手动选择ROI */
    {
        /* 通过鼠标在图像上绘制 ROI */
        std::vector<cv::Point> pts = mouse_selecPtsSet(matImgConfig);
        for (int i = 0; i < 4; i++)
        {
            RB_POINT_S stPt;
            cv::Point cvPt = pts[i];
            stPt.s32X = cvPt.x;
            stPt.s32Y = cvPt.y;
            m_vPoints.push_back(cvPt);
            stPara.vecROIPts[0].push_back(stPt);
        }
    }
    else
    {
        /* 自动 */
        RB_POINT_S stPtTmp[4];
        cv::Point cvPt;

        stPtTmp[0].s32X = INNER_BORDER;
        stPtTmp[0].s32Y = INNER_BORDER;
        stPtTmp[1].s32X = INNER_BORDER;
        stPtTmp[1].s32Y = h - INNER_BORDER;
        stPtTmp[2].s32X = w - INNER_BORDER;
        stPtTmp[2].s32Y = h - INNER_BORDER;
        stPtTmp[3].s32X = w - INNER_BORDER;
        stPtTmp[3].s32Y = INNER_BORDER;

        for (int i = 0; i < 4; i++)
        {
            stPara.vecROIPts[0].push_back(stPtTmp[i]);
            cvPt.x = stPtTmp[i].s32X;
            cvPt.y = stPtTmp[i].s32Y;
            m_vPoints.push_back(cvPt);

        }
    }

    /* 配置算法 */
    ret = RB_CAH_Config(m_AlgoHandle, &stPara);

    return ret;
}


/* 处理每一帧数据 */
int CRbcahExample::process(cv::Mat matImgProcess)
{

    if (matImgProcess.empty())
    {
        std::cout << " The Image is empty, please confirm the image!" << std::endl;
        return RB_FAILURE;
    }




    /* 传入处理的图像 */
    m_matImg = matImgProcess.clone();

    /* 图像赋值 */
    int ret = RB_SUCCESS;
    RB_S32 s32W, s32H;
    s32W = matImgProcess.cols;
    s32H = matImgProcess.rows;
    RB_IMAGE_S stImg;
    stImg.s32W = s32W;
    stImg.s32H = s32H;
    stImg.eFormat = RB_IMAGE_FORMAT_BGR_PACKED;
    stImg.pData = matImgProcess.data;

    /* 处理图像 */
    ret = RB_CAH_Process(m_AlgoHandle, &stImg);
    if (ret != RB_SUCCESS)
    {
        return ret;
    }

    /* 获取结果 */
    ret = RB_CAH_GetResults(m_AlgoHandle, &m_stResults);
    if (ret != RB_SUCCESS)
    {
        return ret;
    }

    m_matImg.copyTo(m_matResult);










    
    return ret;
}

void drawRect(cv::Mat &frame, cv::Rect rect, cv::Scalar colordef)
{
    cv::Point ptCenter, pt1, pt2, pt3, pt4;
    ptCenter.x = rect.x + rect.width / 2;
    ptCenter.y = rect.y + rect.height / 2;

    int w, h;
    w = frame.cols;
    h = frame.rows;

    cv::Scalar color1 = cv::Scalar(0, 200, 64);
    // cv::Scalar color2 = cv::Scalar(0, 0, 255);
    cv::Scalar color2 = colordef;
    cv::Scalar color3 = cv::Scalar(235, 206, 135);

    if (0)
    {
        /* 画十字线 */
        pt1.x = 0,
        pt1.y = ptCenter.y;
        pt2.x = ptCenter.x - rect.width / 2, pt2.y = ptCenter.y;
        cv::line(frame, pt1, pt2, color1, 1, 8, 0);

        pt1.x = ptCenter.x, pt1.y = 0;
        pt2.x = ptCenter.x, pt2.y = ptCenter.y - rect.height / 2;
        cv::line(frame, pt1, pt2, color1, 1, 8, 0);

        pt1.x = ptCenter.x + rect.width / 2, pt1.y = ptCenter.y;
        pt2.x = w - 1, pt2.y = ptCenter.y;
        cv::line(frame, pt1, pt2, color1, 1, 8, 0);

        pt1.x = ptCenter.x, pt1.y = ptCenter.y + rect.height / 2;
        pt2.x = ptCenter.x, pt2.y = h - 1;
        cv::line(frame, pt1, pt2, color1, 1, 8, 0);
    }

    if (1)
    {
        /* 画左上边角 */
        pt1.x = rect.x, pt1.y = rect.y;
        pt2 = pt1, pt2.x += (rect.width >> 3);
        pt3 = pt1, pt3.y += (rect.height >> 3);
        cv::line(frame, pt1, pt2, color2, 4, 8, 0);
        cv::line(frame, pt1, pt3, color2, 4, 8, 0);

        /* 画右下边角 */
        pt1.x = rect.x + rect.width, pt1.y = rect.y + rect.height;
        pt2 = pt1, pt2.x -= (rect.width >> 3);
        pt3 = pt1, pt3.y -= (rect.height >> 3);
        cv::line(frame, pt1, pt2, color2, 4, 8, 0);
        cv::line(frame, pt1, pt3, color2, 4, 8, 0);

        /* 画左上和右下边角 */
        pt1.x = rect.x, pt1.y = rect.y + rect.height;
        pt2 = pt1, pt2.x += (rect.width >> 3);
        pt3 = pt1, pt3.y -= (rect.height >> 3);
        cv::line(frame, pt1, pt2, color2, 4, 8, 0);
        cv::line(frame, pt1, pt3, color2, 4, 8, 0);

        /* 画左上和右下边角 */
        pt1.x = rect.x + rect.width, pt1.y = rect.y;
        pt2 = pt1, pt2.x -= (rect.width >> 3);
        pt3 = pt1, pt3.y += (rect.height >> 3);
        cv::line(frame, pt1, pt2, color2, 4, 8, 0);
        cv::line(frame, pt1, pt3, color2, 4, 8, 0);
    }

    if (1)
    {
        /* 画文字底色 */
        cv::Rect rectText;
        rectText.x = rect.x - 5;
        rectText.y = rect.y - 30;
        rectText.width = 130;
        rectText.height = 30;
        cv::rectangle(frame, rectText, color3, -1, 8, 0);
    }
}

/* 显示结果 */
int CRbcahExample::show_results(int wait_time, float show_ratio)
{

#define WINDOW_NAME "Result"
    int ret = RB_SUCCESS;
    cv::Mat frame = m_matImg.clone();
    cvui::init(WINDOW_NAME);
    double theFontScale = 1;

    /* 打印提示信息 */
    cvui::text(frame, 10, 10, "Result:", theFontScale, 0x20ee17);

    unsigned int theColor;

    /* 画基准位置 */
    cv::Scalar colorgreen = cv::Scalar(0, 255, 0),
            coloryellow = cv::Scalar(0, 255, 255),
            colorred = cv::Scalar(0, 0, 255), showColor;

    /*********************************************/
    for (int i = 0; i < 4; i++)
    {
        // std::cout<<m_vPoints[i]<<std::endl;
        cv::circle(frame, m_vPoints[i], 5, colorgreen, 3, 8, 0);
        cv::line(frame, m_vPoints[i], m_vPoints[(i + 1) % 4], coloryellow, 3, 8, 0);
    }


    for (int i = 0; i < m_stResults.vecPedSet.size(); i++)
    {
        RB_CAH_Ped_S stTmp = m_stResults.vecPedSet[i];
        cv::Rect cvRect;
        cvRect.x = stTmp.stPedBoundingBox.stTopLeft.s32X;
        cvRect.y = stTmp.stPedBoundingBox.stTopLeft.s32Y;
        cvRect.width = stTmp.stPedBoundingBox.stBottomRight.s32X - cvRect.x;
        cvRect.height = stTmp.stPedBoundingBox.stBottomRight.s32Y - cvRect.y;

        std::string theText;
        switch (stTmp.s32labelID)
        {
        case 0:
            theText = "p1";
            showColor = colorgreen;
            break;
        case 1:
            theText = "p2";
            showColor = coloryellow;
            break;
        case 2:
            showColor = colorred;
            theText = "p3";
            break;

        case 3:
            showColor = colorred;
            theText = "Suspected";
            break;

        default:
            showColor = colorred;
            break;
        }

        drawRect(frame, cvRect, showColor);
        cvui::text(frame, cvRect.x - 2, cvRect.y - 25, theText, 0.8, 0xFF1493);
    }

    /*********************************************/
    /* 拷贝结果图像 */
    frame.copyTo(m_matResult);

    /* 更新 UI */
    cvui::update();
    show_ratio=0.5;
    wait_time=1;
    /* 显示图像 */
    cv::Mat matReszShow;
    cv::resize(m_matResult, matReszShow, cv::Size((int)(frame.cols * show_ratio), (int)(frame.rows * show_ratio)));
    cv::imshow(WINDOW_NAME, matReszShow);
    cv::waitKey(wait_time);
#undef WINDOW_NAME

    return ret;
}
