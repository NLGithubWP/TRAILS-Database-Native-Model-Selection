#!/usr/bin/env python
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from download_cifar10 import do_download

if __name__ == '__main__':
    dirpath = '/tmp/'  # need to specify a local directory
    gzfile = dirpath + 'cifar-100-python.tar.gz'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    do_download(dirpath, gzfile, url)
