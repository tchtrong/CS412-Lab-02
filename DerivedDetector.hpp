#ifndef DERIVEDDETECTOR_HPP
#define DERIVEDDETECTOR_HPP

#include "Detector.hpp"
#include "SIFTArgs.hpp"
#include <iostream>

struct HarrisParams final : public DetectorParams
{
    int thresh = 250;
    int block_size = 2;
    int aperture = 3;
    int free_param = 40;
};

struct HarrisData final : DetectorData
{
    using DetectorData::DetectorData;
};

class HarrisDetector final : public Detector
{
public:
    using Detector::create;
    using Detector::Detector;
    void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray()) override;
    const static int MAX_THRESH = 255;
};

struct BlobParams final : public DetectorParams
{
    BlobParams()
    {
        // std::cout << "Blob Color: " << int(this->bp.blobColor) << '\n';
        // std::cout << "Max Circularity: " << this->bp.maxCircularity << ", Min Circularity: " << this->bp.minCircularity << '\n';
        // std::cout << "Max Convexity: " << this->bp.maxConvexity << ", Min Convexity: " << this->bp.minConvexity << '\n';
        // std::cout << "Max InertiaRatio: " << this->bp.maxInertiaRatio << ", Min InertiaRatio: " << this->bp.minInertiaRatio << '\n';
        // std::cout << "minDistBetweenBlobs: " << this->bp.minDistBetweenBlobs << '\n';
        // std::cout << "minRepeatability: " << this->bp.minRepeatability << '\n';
        // std::cout << "thresholdStep: " << this->bp.thresholdStep << '\n';

        // Filter by Color
        this->bp.filterByColor = true;
        // Filter by Area.
        this->bp.filterByArea = true;
        // Filter by Circularity
        this->bp.filterByCircularity = true;
        this->bp.maxCircularity = 1;
        // Filter by Convexity
        this->bp.filterByConvexity = true;
        this->bp.maxConvexity = 1;
        // Filter by Inertia
        this->bp.filterByInertia = true;
        this->bp.maxInertiaRatio = 1;
        this->bp.minInertiaRatio = 0.01;
        // Threshold
        this->bp.maxThreshold = 200;

        this->blob_color = this->bp.blobColor;
        this->blob_area = this->bp.minArea;
        this->blob_circularity = this->bp.minCircularity * 1000;
        this->blob_convexity = this->bp.minConvexity * 1000;
        this->blob_inertia_ratio = this->bp.minInertiaRatio * 1000;
        this->blob_threshold = this->bp.minThreshold;
        this->blob_threshold_step = this->bp.thresholdStep;
    }

    int blob_color;
    int blob_area;
    int blob_circularity;
    int blob_convexity;
    int blob_inertia_ratio;
    int blob_threshold;
    int blob_threshold_step;

    cv::SimpleBlobDetector::Params bp;
};

struct BlobData final : DetectorData
{
    using DetectorData::DetectorData;
};

class BlobDetector final : public Detector
{
public:
    using Detector::create;
    using Detector::Detector;
    void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray()) override;
};

struct DoGParams final : public DetectorParams
{
    SIFTArgs sa;
    int nfeatures = 0;
    int nOctaveLayers = 3;
    int contrastThreshold = 4; // divide for 100
    int edgeThreshold = 10;
    int sigma = 16; // divide for 10
};

struct DoGData final : DetectorData
{
    using DetectorData::DetectorData;
};

class DoGDetector final : public Detector
{
public:
    using Detector::create;
    using Detector::Detector;
    void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray()) override;
};

void handle_dt_params(DetectorData *args);
void run_detector(int, void *);
void create_main_window(DetectorData &args, void (*callback)(int, void *), void *userdata);

#endif