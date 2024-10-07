# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc

import pytest
import torch


def _clear_cache(raise_on_cuda_tensor=False):
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            if not obj.is_cuda:
                continue
            obj_str = str(obj)
            del obj
            if raise_on_cuda_tensor:
                raise AssertionError(f"Found a CUDA tensor in memory after test: {obj_str}")
        except ReferenceError:
            pass


@pytest.fixture(scope="module", autouse=True)
def empty_cuda_cache():
    """Ensure the CUDA memory cache is empty after each test."""
    _clear_cache()
    yield
    _clear_cache(raise_on_cuda_tensor=True)
