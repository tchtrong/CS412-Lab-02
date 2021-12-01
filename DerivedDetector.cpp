#include "DerivedDetector.hpp"
#include "opencv2/highgui.hpp"
#include <bit>

void handle_dt_params(DetectorData *args)
{
    if (auto *args_ = dynamic_cast<HarrisData *>(args))
    {
        auto *params = dynamic_cast<HarrisParams *>(args_->params);
        if (!(params->aperture % 2))
        {
            ++params->aperture;
            cv::setTrackbarPos("Aperture: ", args_->source_window, params->aperture);
        }
    }
    else if (auto *args_ = dynamic_cast<BlobData *>(args))
    {
        auto *params = dynamic_cast<BlobParams *>(args_->params);
        params->bp.blobColor = params->blob_color;
        params->bp.minArea = params->blob_area;
        params->bp.minCircularity = params->blob_circularity / float(1000);
        params->bp.minConvexity = params->blob_convexity / float(1000);
        params->bp.minInertiaRatio = params->blob_inertia_ratio / float(1000);
        params->bp.minThreshold = params->blob_threshold;
        params->bp.thresholdStep = params->blob_threshold_step;
    }
    else if (auto *args_ = dynamic_cast<DoGData *>(args))
    {
        auto *params = dynamic_cast<DoGParams *>(args_->params);
        params->sa = {params->nfeatures,
                      params->nOctaveLayers,
                      params->contrastThreshold / float(100),
                      double(params->edgeThreshold),
                      params->sigma / float(10)};
    }
}

void run_detector(int, void *userdata)
{
    DetectorData *args = std::bit_cast<DetectorData *>(userdata);

    handle_dt_params(args);

    if (!args->src_gray.empty())
    {
        auto detector = Detector::create(args->params);
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(args->src_gray, keypoints);

        cv::Mat gray_circle;
        args->src_gray.copyTo(gray_circle);

        cv::drawKeypoints(args->src_gray, keypoints, gray_circle, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::namedWindow(args->corners_window, cv::WINDOW_NORMAL);
        cv::imshow(args->corners_window, gray_circle);
    }
}

void create_main_window(DetectorData &args, void (*callback)(int, void *), void *userdata)
{
    cv::namedWindow(args.source_window, cv::WINDOW_NORMAL);

    if (auto *params_ = dynamic_cast<HarrisParams *>(args.params))
    {
        auto &params = *params_;
        cv::createTrackbar("Threshold: ", args.source_window, &params.thresh, HarrisDetector::MAX_THRESH, callback, userdata);
        cv::createTrackbar("Block size: ", args.source_window, &params.block_size, 20, callback, userdata);
        cv::setTrackbarMin("Block size: ", args.source_window, 2);
        cv::createTrackbar("Aperture: ", args.source_window, &params.aperture, 7, callback, userdata);
        cv::createTrackbar("Free parameter * 1000: ", args.source_window, &params.free_param, 60, callback, userdata);
    }
    else if (auto *params_ = dynamic_cast<BlobParams *>(args.params))
    {
        auto &params = *params_;
        cv::createTrackbar("Color:", args.source_window, &params.blob_color, 255, callback, userdata);
        cv::createTrackbar("Threshold:", args.source_window, &params.blob_threshold, (int)params.bp.maxThreshold, callback, userdata);
        cv::createTrackbar("Threshold step:", args.source_window, &params.blob_threshold_step, 50, callback, userdata);
        cv::setTrackbarMin("Threshold step:", args.source_window, 1);
        cv::createTrackbar("Area:", args.source_window, &params.blob_area, (int)params.bp.maxArea, callback, userdata);
        cv::createTrackbar("Circularity * 1000:", args.source_window, &params.blob_circularity, int(params.bp.maxCircularity) * 1000, callback, userdata);
        cv::createTrackbar("Convexity * 1000:", args.source_window, &params.blob_convexity, int(params.bp.maxConvexity) * 1000, callback, userdata);
        cv::createTrackbar("Inertia Ratio * 1000:", args.source_window, &params.blob_inertia_ratio, int(params.bp.maxInertiaRatio) * 1000, callback, userdata);
    }
    else if (auto *params_ = dynamic_cast<DoGParams *>(args.params))
    {
        auto &params = *params_;
        cv::createTrackbar("nfeatures", args.source_window, &params.nfeatures, 10, callback, userdata);
        cv::createTrackbar("nOctaveLayers", args.source_window, &params.nOctaveLayers, 10, callback, userdata);
        cv::createTrackbar("contrastThreshold * 100", args.source_window, &params.contrastThreshold, 10, callback, userdata);
        cv::createTrackbar("edgeThreshold", args.source_window, &params.edgeThreshold, 100, callback, userdata);
        cv::createTrackbar("sigma * 10", args.source_window, &params.sigma, 20, callback, userdata);
    }
}