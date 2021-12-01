#include "DerivedDescriptor.hpp"

void handle_dst_ext_params(Descriptor *dst)
{
    if (auto *args_ = dynamic_cast<SIFTDesExt *>(dst))
    {
        auto *params = dynamic_cast<SIFTDesExtParams *>(args_->params);
        params->sa = {params->nfeatures,
                      params->nOctaveLayers,
                      params->contrastThreshold / float(100),
                      double(params->edgeThreshold),
                      params->sigma / float(10)};
    }
}