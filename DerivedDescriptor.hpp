#ifndef DERIVEDDESCRIPTOR_HPP
#define DERIVEDDESCRIPTOR_HPP

#include "Descriptor.hpp"
#include "SIFTArgs.hpp"

struct SIFTDesExtParams : public DescriptorParams
{
    SIFTArgs sa;
    int nfeatures = 0;
    int nOctaveLayers = 3;
    int contrastThreshold = 4; // divide for 100
    int edgeThreshold = 10;
    int sigma = 16; // divide for 10
};

class SIFTDesExt final : public Descriptor
{
public:
    using Descriptor::create;
    using Descriptor::Descriptor;
    void compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors) override;
};

class LBPDesExt final : public Descriptor
{
public:
    using Descriptor::create;
    using Descriptor::Descriptor;
    void compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors) override;
};

void handle_dst_ext_params(Descriptor *dst);

#endif