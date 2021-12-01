#include "DerivedDetector.hpp"

DetectorParams::~DetectorParams() {}

Detector::~Detector() {}

cv::Ptr<Detector> Detector::create(DetectorParams *params)
{
    if (auto *new_params = dynamic_cast<HarrisParams *>(params))
    {
        return cv::Ptr<HarrisDetector>(new HarrisDetector(params));
    }
    else if (auto *new_params = dynamic_cast<BlobParams *>(params))
    {
        return cv::Ptr<BlobDetector>(new BlobDetector(params));
    }
    else if (auto *new_params = dynamic_cast<DoGParams *>(params))
    {
        return cv::Ptr<DoGDetector>(new DoGDetector(params));
    }
    return nullptr;
}

DetectorData::~DetectorData() {}
