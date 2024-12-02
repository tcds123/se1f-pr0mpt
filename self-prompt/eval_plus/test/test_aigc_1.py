# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import random
import string
import sys
import time
import uuid

import requests

CHAR_LIST = []
[[CHAR_LIST.append(e) for e in string.ascii_letters] for i in range(0, 2)]
[[CHAR_LIST.append(e) for e in string.ascii_letters] for i in range(0, 2)]
[[CHAR_LIST.append(e) for e in string.digits] for i in range(0, 2)]


# generate nonce
def get_chars(length):
    """get the random chars

    :param length: lenght of the string
    :returns: string

    """
    random.shuffle(CHAR_LIST)
    return ''.join(CHAR_LIST[0:length])


def queryByOpenAPI(queryDict, secretKey, publicKey, remoteUrl):
    nonce = get_chars(8)
    timestamp = str(int(time.time() * 1000))
    unionStr = nonce + timestamp + json.dumps(queryDict, ensure_ascii=False)
    # print("unionStr:", unionStr)
    signature = base64.b64encode(
        hmac.new(secretKey.encode('utf-8'), unionStr.encode('utf-8'), digestmod=hashlib.sha256).digest()).decode('utf-8')
    # print("Signature: ", signature)
    headers = {
        "Content-Type": "application/json",
        "PublicKey": publicKey,
        "Nonce": nonce,
        "Timestamp": timestamp,
        "Signature": signature
    }
    data = requests.request("POST", remoteUrl, headers=headers,
                            data=str(json.dumps(queryDict, ensure_ascii=False)).encode('utf-8')).json()
    # print("response data:", data)
    return data


# API URLs
final_url = "https://aibuilder.midea.com/aigc-openapi/v1/open/qa"

if __name__ == "__main__":
    # get uuid
    my_uuid = str(uuid.uuid1()).replace('-', '')

    ## 如下两个ID在"应用设置"中看到，替换为自己应用的ID
    pubkey = "9b3adf547b894c28b586b68a6cf15d1e"
    secret = "a33f21701e3848b2aef57a1a4bfa4de5"

    query = '''
以下是上文和下文和组件模版，根据上文和下文，对组件模版属性进行完善。你只需返回完善后的组件模版，不要返回上文和下文。
{
上文:
<template>
  <md-container>
    <!-- <md-header height="48px">
      <apaas-page-header project-name="123456" />
    </md-header> -->
    <md-container>
      <md-aside width="200px">
        <md-scrollbar>
          <md-menu>
            <md-menu-item 
              v-for="item in routes" 
              :key="item.path" 
              :index="item.path" 
              @click="handleClickMenu"
            >
              {{ item.meta?.label }}
            </md-menu-item>
          </md-menu>
        </md-scrollbar>
      </md-aside>
      <md-main class="full-height">
        <div class="example-main-container">
          <apaas-form widget-id="apaasForm" :model="formData">
            <router-view />
          </apaas-form>
        </div>
      </md-main>
},
{
下文:
         </md-container>
  </md-container>
</template>

<script setup lang="ts">
import { MdContainer, MdAside, MdMain, MdScrollbar, MdMenu, MdMenuItem, MenuItemRegistered } from 'mdesign3'
import { routes } from '@/router'
import { useRouter } from 'apaas-kit-common/utils/router';
import { provide, reactive } from 'vue';
import ApaasForm from '@/packages/Form/index.vue'
import dataSources from './dataSource.json'
import { setProjectDataSource } from '..';

const router = useRouter()

const handleClickMenu = (e: MenuItemRegistered) => {
  router.push({
    path: e.index
  })
}

setProjectDataSource(dataSources)

const formData = reactive({})
</script>

<style scoped lang="scss">
.example-main-container {
  width: 100%;
  height: 100%;
  overflow-y: auto;
}
.full-height {
  height: 100vh;
}
</style>
},
{
模版:
<md-button type="primary">Primary</md-button> 
}
'''
    payload = {
        ## 可以替换为自己的IMIP名字
        "userId": "zongwx1",
        "requestId": my_uuid,
        "query": query
    }
    result = queryByOpenAPI(payload, secret, pubkey, final_url)
    print(result)

    sys.exit(0)

