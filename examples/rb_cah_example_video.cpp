

#include <iostream>
#include "rb_cah_example_utils.h"

#define GAP_FRM_NUM 1   /* 隔帧 */
#define WAIT_TIME   1  /* 等待时间 */

using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " video path" << std::endl;
        return -1;
    }

    /* 读取视频 */
    const string source = argv[1];
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cout << "Could not open the video " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
    {
        return -1;
    }

    RB_S32 s32FrmIDOut = 0;
    /* 写视频 */
    cv::VideoWriter outputVideo;
    string::size_type pAt = source.find_last_of('.');          // Find extension point
    const string NAME = source.substr(0, pAt) + "_result.mp4"; // Form the new name with container
    int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));       // Get Codec Type- Int form
    outputVideo.open(NAME, ex, 8, frame.size(), true);

    /* 初始化算法 */
    CRbcahExample *example = new CRbcahExample(frame);

    /* 配置算法 */
    example->config(frame, true);

    while (1)
    {
        /* 读取帧 */
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        s32FrmIDOut++;

        if (s32FrmIDOut % GAP_FRM_NUM != 0)
        {
            continue;
        }

        /* 处理帧 */
        example->process(frame);

        /* 显示结果 */
        example->show_results(10, 0.5);

        /* 保存效果视频 */
        outputVideo << example->m_matResult;

        int key = cv::waitKey(WAIT_TIME);
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    /* 释放资源 */
    delete example;
    cap.release();
    outputVideo.release();
    cv::destroyAllWindows();

    return 0;
}