#include "Descriptor.hpp"

DescriptorParams::~DescriptorParams() {}

Descriptor::~Descriptor() {}

cv::Ptr<Descriptor> Descriptor::create(DescriptorParams *params)
{
    return nullptr;
}