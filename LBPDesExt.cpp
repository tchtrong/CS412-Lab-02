#include "DerivedDescriptor.hpp"
#include <algorithm>

void LBPDesExt::compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors)
{
    descriptors.create(cv::Size(256, keypoints.size()), CV_32F);
    cv::Mat des = descriptors.getMat();
    des.setTo(0);

    cv::Mat src = image.getMat();
    int sx = image.rows();
    int sy = image.cols();

    for (size_t c = 0; c < keypoints.size(); ++c)
    {
        auto [y, x] = keypoints[c].pt;
        int top = std::max(int(x - 15), 0);
        int down = std::min(int(x + 15), sx - 1);
        int left = std::max(int(y - 15), 0);
        int right = std::min(int(y + 15), sy - 1);
        cv::Mat tmp = cv::Mat::zeros(cv::Size(right - left - 1, down - top - 1), CV_8U);
        for (int i = top + 1; i < down - 1; ++i)
        {
            for (int j = left + 1; j < right - 1; ++j)
            {
                unsigned char center = src.at<unsigned char>(i, j);
                unsigned char code = 0;
                code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
                code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
                code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
                code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
                code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
                code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
                code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
                code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
                tmp.at<unsigned char>(i - top - 1, j - left - 1) = code;
            }
        }
        for (int i = top + 1; i < down - 1; ++i)
        {
            for (int j = left + 1; j < right - 1; ++j)
            {
                unsigned char bin = tmp.at<unsigned char>(i - top - 1, j - left - 1);
                des.at<float>(c, bin) += 1;
            }
        }
    }
}