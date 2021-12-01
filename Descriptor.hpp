#ifndef DESCRIPTOR_HPP
#define DESCRIPTOR_HPP

#include "opencv2/features2d.hpp"

class DescriptorParams
{
public:
    virtual ~DescriptorParams() = 0;
};

class Descriptor : public cv::DescriptorExtractor
{
public:
    Descriptor(DescriptorParams *params) : params{params} {}
    virtual ~Descriptor() = 0;
    static cv::Ptr<Descriptor> create(DescriptorParams *params);
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors) = 0;
    DescriptorParams *params;
};

#endif
