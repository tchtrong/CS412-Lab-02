#include "DerivedDetector.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void BlobDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask)
{
    if (auto *params_ = dynamic_cast<BlobParams *>(this->params))
    {
        auto detector = cv::SimpleBlobDetector::create(params_->bp);
        detector->detect(image, keypoints, mask);
    }
}