#include "Matcher.hpp"
#include <bit>
#include "opencv2/highgui.hpp"
#include <algorithm>

int MatcherData::count = 0;
std::vector<DetectorData *> MatcherData::lst_has = std::vector<DetectorData *>(2);
Descriptor *MatcherData::dst = nullptr;
std::vector<std::vector<cv::KeyPoint>> MatcherData::kps = std::vector<std::vector<cv::KeyPoint>>(2);
std::vector<cv::Mat> MatcherData::descriptors = std::vector<cv::Mat>(2);
std::vector<cv::DMatch> MatcherData::matches = std::vector<cv::DMatch>();
int MatcherData::k = 10;
int MatcherData::n_shown = 200;
const cv::String MatcherData::winname = "Good Matches";

void create_match_window(Descriptor &args, void (*callback)(int, void *), void *userdata)
{
    cv::namedWindow(MatcherData::winname, cv::WINDOW_NORMAL);
    if (auto *params_ = dynamic_cast<SIFTDesExtParams *>(args.params))
    {
        // int  	nfeatures = 0,
        // int  	nOctaveLayers = 3,
        // double  	contrastThreshold = 0.04,
        // double  	edgeThreshold = 10,
        // double  	sigma = 1.6
        auto &params = *params_;
        cv::createTrackbar("nfeatures", MatcherData::winname, &params.nfeatures, 10, callback, userdata);
        cv::createTrackbar("nOctaveLayers", MatcherData::winname, &params.nOctaveLayers, 10, callback, userdata);
        cv::createTrackbar("contrastThreshold * 100", MatcherData::winname, &params.contrastThreshold, 10, callback, userdata);
        cv::createTrackbar("edgeThreshold", MatcherData::winname, &params.edgeThreshold, 100, callback, userdata);
        cv::createTrackbar("sigma * 10", MatcherData::winname, &params.sigma, 20, callback, userdata);
    }
    cv::createTrackbar("Number of matches to be shown:", MatcherData::winname, &MatcherData::n_shown, 10, show_matches, userdata);
}

void show_matches(int, void *)
{
    cv::Mat img_matches;
    std::vector<cv::DMatch> shown;
    auto to_shown = std::min(MatcherData::n_shown, int(MatcherData::matches.size()));

    if (!MatcherData::matches.empty())
    {
        std::copy_n(MatcherData::matches.begin(), to_shown, std::back_inserter(shown));
    }

    cv::drawMatches(MatcherData::lst_has[0]->src, MatcherData::kps[0], MatcherData::lst_has[1]->src, MatcherData::kps[1], shown, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow(MatcherData::winname, img_matches);
}

void run_matcher()
{
    if (!MatcherData::descriptors[0].empty() && !MatcherData::descriptors[1].empty())
    {
        if (MatcherData::kps[0].size() >= MatcherData::k && MatcherData::kps[1].size() >= MatcherData::k)
        {
            auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(MatcherData::descriptors[0], MatcherData::descriptors[1], knn_matches, MatcherData::k);

            MatcherData::matches.clear();
            for (auto &knn_match : knn_matches)
            {
                std::move(knn_match.begin(), knn_match.end(), std::back_inserter(MatcherData::matches));
            }
            std::sort(MatcherData::matches.begin(), MatcherData::matches.end(), [](auto &a, auto &b)
                      { return a.distance < b.distance; });

            cv::setTrackbarMax("Number of matches to be shown:", MatcherData::winname, MatcherData::matches.size());
            cv::setTrackbarPos("Number of matches to be shown:", MatcherData::winname, MatcherData::matches.size() / 2);
        }
    }
    show_matches(0, nullptr);
}

void run_detector_matcher(int, void *userdata)
{
    MatcherData *md_ = std::bit_cast<MatcherData *>(userdata);
    MatcherData &md = *md_;
    auto idx = md.idx;

    auto *dt = MatcherData::lst_has[idx];
    handle_dt_params(dt);

    if (!dt->src_gray.empty())
    {
        auto detector = Detector::create(dt->params);
        detector->detect(dt->src_gray, MatcherData::kps[idx]);

        cv::Mat gray_circle;
        dt->src_gray.copyTo(gray_circle);

        cv::drawKeypoints(dt->src_gray, MatcherData::kps[idx], gray_circle, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imshow(dt->corners_window, gray_circle);
    }
    md.dst->compute(dt->src_gray, md.kps[idx], md.descriptors[idx]);
    run_matcher();
}

void run_dst_ext(int, void *)
{
    handle_dst_ext_params(MatcherData::dst);
    MatcherData::dst->compute(MatcherData::lst_has[0]->src_gray, MatcherData::kps[0], MatcherData::descriptors[0]);
    MatcherData::dst->compute(MatcherData::lst_has[1]->src_gray, MatcherData::kps[1], MatcherData::descriptors[1]);
    run_matcher();
}