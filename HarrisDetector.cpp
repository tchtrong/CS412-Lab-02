#include "DerivedDetector.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
// #include <cmath>

void HarrisDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask)
{
    cv::Mat dst;
    auto *params_ = dynamic_cast<HarrisParams *>(this->params);
    cv::cornerHarris(image, dst, params_->block_size, params_->aperture, params_->free_param / float(1000));

    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, MAX_THRESH, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    keypoints.clear();

    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if (dst_norm.at<float>(i, j) > params_->thresh)
            {
                keypoints.emplace_back(j, i, int(1 / dst_norm.at<float>(i, j) * MAX_THRESH * 5));
            }
        }
    }
    auto [x, y] = dst.size();
    if (float(keypoints.size()) / (x * y) > 0.8)
    {
        keypoints.clear();
    }
}
