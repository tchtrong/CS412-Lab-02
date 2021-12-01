#include "DerivedDetector.hpp"
#include <iostream>

void DoGDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask)
{
    if (auto *params_ = dynamic_cast<DoGParams *>(this->params))
    {
        auto detector = cv::SIFT::create(params_->sa.nfeatures,
                                         params_->sa.nOctaveLayers,
                                         params_->sa.contrastThreshold,
                                         params_->sa.edgeThreshold,
                                         params_->sa.sigma);
        detector->detect(image, keypoints, mask);
    }
}