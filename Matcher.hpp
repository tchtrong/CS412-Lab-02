#ifndef MATCHER_HPP
#define MATCHER_HPP

#include "opencv2/core.hpp"
#include "DerivedDescriptor.hpp"
#include "DerivedDetector.hpp"

struct MatcherData
{
    MatcherData() : idx{count++} {}

    static std::vector<DetectorData *> lst_has;
    static Descriptor *dst;
    static std::vector<std::vector<cv::KeyPoint>> kps;
    static std::vector<cv::Mat> descriptors;
    static std::vector<cv::DMatch> matches;
    static int k;
    static int n_shown;

    const static cv::String winname;

    const int idx;

private:
    static int count;
};

void create_match_window(Descriptor &args, void (*callback)(int, void *), void *userdata);
void run_detector_matcher(int, void *userdata);
void run_dst_ext(int, void *userdata);
void run_matcher();
void show_matches(int, void*);

#endif