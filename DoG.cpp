#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "DerivedDetector.hpp"

#include <iostream>

const cv::String keys =
    "{help h usage ?    |     |   Print help message}"
    "{@input            |     |   input image}";

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!(parser.get<cv::String>("@input").compare("")))
    {
        cv::VideoCapture cam{0};

        DoGParams params;
        DoGData args{&params};
        create_main_window(args, run_detector, &args);

        while (true)
        {
            cam.read(args.src);

            if (!args.src.empty())
            {
                cv::cvtColor(args.src, args.src_gray, cv::COLOR_BGR2GRAY);
                run_detector(0, &args);
            }

            cv::imshow(args.source_window, args.src);

            if (cv::waitKey(1) != -1)
            {
                break;
            }
        }
        cam.release();
    }
    else
    {
        DoGParams params;
        DoGData args{&params};

        args.src = cv::imread(parser.get<cv::String>("@input"));

        if (args.src.empty())
        {
            std::cout << "Could not open or find the image" << parser.get<cv::String>("@input") << "\n";
            parser.printErrors();
            return -1;
        }

        create_main_window(args, run_detector, &args);

        cv::cvtColor(args.src, args.src_gray, cv::COLOR_BGR2GRAY);
        run_detector(0, &args);

        cv::imshow(args.source_window, args.src);

        cv::waitKey();
    }

    cv::destroyAllWindows();

    return 0;
}