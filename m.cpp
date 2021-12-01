#include "Matcher.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <bit>

// void corner_harris(int, void *);
// void harris_sift(int, void *userdata);
const cv::String HARRIS = "harris";
const cv::String BLOB = "blob";
const cv::String DOG = "dog";
const cv::String SIFT = "sift";
const cv::String LBP = "lbp";

const cv::String keys =
    "{ help h       |                       | Print help message. }"
    "{ @detector    | harris                | Detector}"
    "{ @descriptor  | lbp                  | Descriptor}"
    "{ @input1      | box.png               | Path to input image 1. }"
    "{ @input2      | box_in_scene.png      | Path to input image 2. }";

void ResizeWithAspectRatio(cv::Mat image, int width, int height, cv::InterpolationFlags flag = cv::INTER_AREA)
{
    cv::Size dim;

    auto [w, h] = image.size();

    auto sw = w / float(width);
    auto sh = h / float(height);
    
    cv::Mat tmp = image.clone();
    cv::resize(tmp, image, dim, 0, 0, cv::INTER_AREA);
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    auto detector = parser.get<cv::String>("@detector");
    auto descriptor = parser.get<cv::String>("@descriptor");

    MatcherData img1;
    MatcherData img2;

    DetectorParams *dtp1;
    DetectorParams *dtp2;

    DetectorData *dtd1;
    DetectorData *dtd2;

    DescriptorParams *dstp;
    Descriptor *dst;

    if (detector == HARRIS)
    {
        dtp1 = new HarrisParams();
        dtp2 = new HarrisParams();

        dtd1 = new HarrisData(dtp1);
        dtd2 = new HarrisData(dtp2);
    }
    else if (detector == BLOB)
    {
        dtp1 = new BlobParams();
        dtp2 = new BlobParams();

        dtd1 = new BlobData(dtp1);
        dtd2 = new BlobData(dtp2);
    }
    else if (detector == DOG)
    {
        dtp1 = new DoGParams();
        dtp2 = new DoGParams();

        dtd1 = new DoGData(dtp1);
        dtd2 = new DoGData(dtp2);
    }

    if (descriptor == SIFT)
    {
        dstp = new SIFTDesExtParams();
        dst = new SIFTDesExt(dstp);
    }
    else if (descriptor == LBP)
    {
        dst = new LBPDesExt(dstp);
    }

    img1.lst_has[0] = dtd1;
    img1.lst_has[1] = dtd2;
    img1.dst = dst;

    dtd1->source_window = "Image 1";
    dtd2->source_window = "Image 2";

    dtd1->corners_window = "Image 1's keypoints";
    dtd2->corners_window = "Image 2's keypoints";

    create_main_window(*dtd1, run_detector_matcher, &img1);
    create_main_window(*dtd2, run_detector_matcher, &img2);

    img1.lst_has[0]->src = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input1")));
    if (img1.lst_has[0]->src.empty())
    {
        std::cout << "Could not open or find the image: " << parser.get<cv::String>("@input1") << '\n';
        return -1;
    }
    // ResizeWithAspectRatio(img1.lst_has[0]->src, 640, 320);
    cv::imshow(img1.lst_has[0]->source_window, img1.lst_has[0]->src);

    img2.lst_has[1]->src = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input2")));
    if (img2.lst_has[1]->src.empty())
    {
        std::cout << "Could not open or find the image: " << parser.get<cv::String>("@input2") << '\n';
        return -1;
    }
    // ResizeWithAspectRatio(img1.lst_has[0]->src, 640, 320);
    cv::imshow(img2.lst_has[1]->source_window, img2.lst_has[1]->src);

    cv::cvtColor(img1.lst_has[0]->src, img1.lst_has[0]->src_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img1.lst_has[1]->src, img1.lst_has[1]->src_gray, cv::COLOR_BGR2GRAY);

    create_match_window(*MatcherData::dst, run_dst_ext, nullptr);

    run_detector_matcher(0, &img1);
    run_detector_matcher(0, &img2);
    // run_dst_ext(0, nullptr);

    cv::waitKey(0);

    delete dtp1;
    delete dtp2;
    delete dtd1;
    delete dtd2;
    if (dstp)
    {
        delete dstp;
    }
    delete dst;

    return 0;
}