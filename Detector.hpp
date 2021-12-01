#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include "opencv2/features2d.hpp"

struct DetectorParams
{
    virtual ~DetectorParams() = 0;
};

struct DetectorData
{
    DetectorData(DetectorParams *params) : params{params} {}
    virtual ~DetectorData() = 0;

    cv::String source_window = "Source image";
    cv::String corners_window = "Corners detected";

    cv::Mat src;
    cv::Mat src_gray;

    DetectorParams *params;
};

class Detector : public cv::FeatureDetector
{
public:
    Detector(DetectorParams *params) : params{params} {}
    virtual ~Detector() = 0;
    static cv::Ptr<Detector> create(DetectorParams *params);
    virtual void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray()) = 0;
    DetectorParams *params;
};

#endif