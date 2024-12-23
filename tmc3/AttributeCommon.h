/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2019, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <stdint.h>
#include <vector>

#include "entropy.h"
#include "hls.h"
#include "PCCTMC3Common.h"

namespace pcc {

//============================================================================

class AttributeContexts {
public:
  void reset();

protected:
  AdaptiveBitModel ctxRunLen[5];
  AdaptiveBitModel ctxCoeffGtN[2][7];
  AdaptiveBitModel ctxCoeffRemPrefix[2][12];
  AdaptiveBitModel ctxCoeffRemSuffix[2][12];
};

//----------------------------------------------------------------------------

inline void
AttributeContexts::reset()
{
  this->~AttributeContexts();
  new (this) AttributeContexts;
}

//============================================================================
struct LayerGroupSlicingParams;

//============================================================================

struct AttributeLods {
  // Indicates if the generated LoDs are compatible with the provided aps
  bool isReusable(
    const AttributeParameterSet& aps, const AttributeBrickHeader& abh) const;

  bool empty() const { return numPointsInLod.empty(); };

  std::vector<uint32_t> generate(
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    int geom_num_points_minus1,
    int minGeomNodeSizeLog2,
    const PCCPointSet3& cloud,
    const AttributeInterPredParams& attrInterPredParams,
    int maxLevel = 0,
    bool isGSUnit = false);

  void generateLod(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  int geom_num_points_minus1,
  int minGeomNodeSizeLog2,
  const PCCPointSet3& cloud,
  int maxLevel = 0);

  //used only for weight_adjustment_method==1
  //++++++++++++++++++++++++++++++++++++++++++++++
  void generate_forFullLayerGroupSlicingEncoder(
	  const AttributeParameterSet& aps,
	  const AttributeBrickHeader& abh,
	  int geom_num_points_minus1,
	  int minGeomNodeSizeLog2,
	  const PCCPointSet3& cloud,
	  const AttributeInterPredParams& attrInterPredParams,
	  bool layerGroupEnabledFlag,
	  int rootNodeSizeLog2,
	  int rootNodeSizeLog2_coded,
	  std::vector<std::vector<std::vector<uint32_t>>> dcmNodesIdx,
	  std::vector<int> numLayersPerLayerGroup,
	  std::vector<std::vector<Vec3<int>>> subgrpBboxOrigin,
	  std::vector<std::vector<Vec3<int>>> subgrpBboxSize,
	  std::vector<std::vector<int>> sliceSelectionIndicationFlag,
	  std::vector<int>& numberOfPointsPerLodPerSubgroups);
  //++++++++++++++++++++++++++++++++++++++++++++++

  void predictFromParent(
  const PCCPointSet3& cloud,
  const PCCPointSet3& cloudParent);

  std::vector<PCCPredictor> predictors;
  std::vector<uint32_t> numPointsInLod;
  std::vector<uint32_t> indexes;

  std::vector<int> numRefNodesInTheSameSubgroup;

  std::vector<uint32_t> indexesRef;
  std::vector<uint32_t> numPointsInLodRef;
  
  bool canonical_lod_sampling_enabled_flag;

private:
  // This is the aps that was used to generate the LoDs.  It is used to check
  // if the generated LoDs are reusable.
  AttributeParameterSet _aps;

  // This is the abh that was used to generate the LoDs.  It is used to check
  // if the generated LoDs are reusable.
  AttributeBrickHeader _abh;
};

//============================================================================

bool predModeEligibleColor(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexes,
  const PCCPredictor& predictor,
  const AttributeInterPredParams& attrInterPredParams);

bool predModeEligibleRefl(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexes,
  const PCCPredictor& predictor
  , const AttributeInterPredParams& attrInterPredParams);

//============================================================================

}  // namespace pcc
