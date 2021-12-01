#include "DerivedDescriptor.hpp"

void SIFTDesExt::compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors)
{
    if (auto *params_ = dynamic_cast<SIFTDesExtParams *>(this->params))
    {
        auto &params__ = *params_;
        auto de = cv::SIFT::create();
        de->compute(image, keypoints, descriptors);
        // params__.sa.nfeatures, params__.sa.nOctaveLayers, params__.sa.contrastThreshold, params__.sa.edgeThreshold, params__.sa.sigma
    }
}