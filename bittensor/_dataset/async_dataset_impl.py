##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import bittensor
import streamlit as st
import numpy as np
import asyncio
import aiohttp
import json
import os

from fsspec.asyn import AsyncFileSystem, sync, sync_wrapper

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
##########################
##### Get args ###########
##########################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
# bittensor.wallet.add_args(parser)
# bittensor.logging.add_args(parser)
# bittensor.subtensor.add_args(parser)
# config = bittensor.config(parser = parser)
# config.wallet.name = 'const'
# config.wallet.hotkey = 'Tiberius'
##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.


"""

Background Actor for Message Brokers Between Quees

"""


class Dataset():
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self, loop=None):
        
        # Used to retrieve directory contentx
        self.ipfs_url = 'http://global.ipfs.opentensor.ai/api/v0'
        self.dataset_dir = 'http://global.ipfs.opentensor.ai/api/v0/cat' 
        self.text_dir = 'http://global.ipfs.opentensor.ai/api/v0/object/get'
        self.mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
        # Used when current corpus has been exhausted
        self.refresh_corpus = False
        self.loop = self.get_event_loop()
        # self.queue_server = AsyncQueueServer(loop=loop)  
 
        # object_links = loop.run_until_complete(self.ipfs_get_object(self.hash_table[1]))['Links']
        

        # st.write(self.datasets)
        self.dataset_hashes = self.async_run(self.get_dataset_hashes())
        folder_hashes = self.async_run(self.get_folder_hashes(self.dataset_hashes[0]))
        st.write(folder_hashes)
        for f in folder_hashes:
            text_hashes = self.async_run(self.get_text_hashes(f))
            st.write(text_hashes)
        # st.write(text_hashes)

        # st.write(self.async_run(self.get_dataset_samples(self.dataset_hashes[1])))
        # st.write(self.dataset_hashes[0])
        # folder_hashes = self.async_run(self.get_folder_hashes(self.dataset_hashes[0]))
        # text_hashes = self.async_run(self.get_text_hashes(folder_hashes[0]))
        # text = self.async_run(asyncio.gather(*[self.get_text(t) for t in text_hashes]))
        # st.write(text)

        # st.write(self.async_run(self.get_dataset_samples(self.dataset_hashes[2])))
            
        # st.write(self.async_run(self.get_text(self.dataset_hashes[0]) ))


        # object_links = loop.run_until_complete(asyncio.gather(*[await self.get_text(file_meta) for file_meta in self.hash_table]))
        # object_links[0]
        # st.write(object_links)
        # objects = asyncio.run(self.ipfs_object_get(self.hash_table[1], timeout=2))['Links']        
        # objects = asyncio.run(self.ipfs_object_get(objects[0], timeout=2))
        # st.write(objects)

    async def get_dataset_samples(self, file_meta, num_samples=50, chunk_size=1024):
        links = await self.get_links(file_meta)
        sample_hashes = []
        folder_hashes = []
        # unfinished_links = [ for link in links]
        for link in links:
            finished_links, unfinished_links = await asyncio.wait(unfinished_links)
            # st.write(st.write(link))
            # link = self.ipfs_post('object/get', params={'arg':link['Hash']})

            for job_link in finished_links:
                async for data in res.content.iter_chunked(1024):  
                    hashes = ['['+h + '}]'for h in data.decode().split('},')]
                    folder_hashes = []
                    for i in range(len(hashes)-1):
                        try:
                            folder_hashes += [json.loads(hashes[i+1][1:-1])]
                        except json.JSONDecodeError:
                            pass
                        # hashes[i] =bytes('{'+ hashes[i+1] + '}')
                    if len(folder_hashes) == 0:
                        continue
                    unfinished = [self.api_post('cat', params={'arg':f['Hash']}) for f in folder_hashes]
                    while len(unfinished) > 0 :
                        finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
                        for job in finished:
                            sample_res = await job

                            async for data in sample_res.content.iter_chunked(256):  
                                hashes = ['['+h + '}]'for h in data.decode().split('},')]
                                for i in range(len(hashes)-1):
                                    try:
                                        sample_hashes += [json.loads(hashes[i+1][1:-1])]
                                    except json.JSONDecodeError:
                                        pass

                                st.write(len(sample_hashes))

                                if len(sample_hashes) >= num_samples:
                                    return sample_hashes
                    # hashes[i] =bytes('{'+ hashes[i+1] + '}')



        # text = self.async_run(asyncio.gather(*[self.get_text(t) for t in text_hashes]))

        return text_hashes
    async def get_dataset_hashes(self):
        mountain_meta = {'Name': 'mountain', 'Folder': 'meta_data', 'Hash': self.mountain_hash}
        response = await self.api_post( url=f'{self.ipfs_url}/object/get',  params={'arg': mountain_meta['Hash']}, reuturn_json= True)
        st.write(response)
        response = response.get('Links', None)
        return response


    @property
    def dataset2size(self):
        return {k:v['Size'] for k,v in self.dataset2hash.items()}


    @property
    def datasets(self):
        return list(self.dataset2hash.keys())
    @property
    def dataset2hash(self):
        if not hasattr(self, 'dataset_table'):
            self.dataset_hashes = self.async_run(self.get_dataset_hashes())
        return {v['Name'].replace('.txt', '') :v for v in self.dataset_hashes}
    



    async def get_folder_hashes(self, file_meta, num_folders = 10, chunk_size=512):
        links = await self.get_links(file_meta)

        unfinished = [self.loop.create_task(self.api_post(self.ipfs_url+'/object/get', params={'arg':link['Hash']})) for link in links][:num_folders]
        folder_hashes = []
        while len(unfinished)>0:
            finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            # st.write(len(finished), len(unfinished))
            for res in await asyncio.gather(*finished):

                folder_hashes.extend(res.get('Links'))
                # async for data in res.content.iter_chunked(chunk_size):  
                #     hashes = ['['+h + '}]'for h in data.decode().split('},')]
                #     for i in range(len(hashes)-1):
                #         try:
                #             folder_hashes += [json.loads(hashes[i+1][1:-1])]
                #         except json.JSONDecodeError:
                #             pass

                #         if len(folder_hashes) >= num_folders:
                #             return folder_hashes

        return folder_hashes
        # for link in links:
        #     total_data = b''
        #     async for data in res.content.iter_chunked(1024):  
        #         total_data += data
        #         st.write(data)
        #         hashes = ['['+h + '}]'for h in data.decode().split('},')]
        #         decoded_hashes = []
        #         for i in range(len(hashes)-1):
        #             try:
        #                 decoded_hashes += [json.loads(hashes[i+1][1:-1])]
        #             except json.JSONDecodeError:
        #                 pass
        #             # hashes[i] =bytes('{'+ hashes[i+1] + '}')
        #         if len(decoded_hashes) >= num_hashes:
        #             return decoded_hashes


    async def get_text_hashes(self, file_meta, chunk_size=1024, num_hashes=10):

        try:
            res = await self.api_post(f'{self.ipfs_url}/cat', params={'arg':file_meta['Hash']}, return_json=False, num_chunks=10)
            return res
        except KeyError:
            return []
        decoded_hashes = []
        async for data in res.content.iter_chunked(chunk_size):  
            hashes = ['['+h + '}]'for h in data.decode().split('},')]
            for i in range(len(hashes)-1):
                try:
                    decoded_hashes += [json.loads(hashes[i+1][1:-1])]
                except json.JSONDecodeError:
                    pass

                if len(decoded_hashes) >= num_hashes:
                    return decoded_hashes
                # hashes[i] =bytes('{'+ hashes[i+1] + '}')



    async def get_text(self, file_meta, chunk_size=1024, num_chunks=2):
        res = await self.ipfs_post('cat', params={'arg':file_meta['Hash']})
        total_data = b''
        cnt = 0
        async for data in res.content.iter_chunked(chunk_size):  
            total_data += data
            cnt+= 1
            if cnt >= num_chunks:
                break
        
        return total_data

    

        
    async def get_links(self, file_meta, resursive=False, **kwargs):
        response = await self.api_post( url=f'{self.ipfs_url}/object/get',  params={'arg': file_meta['Hash']}, reuturn_json= True)
        response_links = response.get('Links', [])
        return response_links

    async def get_file_metas(self, file_meta):
        response = await self.ipfs_get_object(file_meta)
        return response
    async def get_files(self, file_meta, limit=1000, **kwargs):

        file_meta_list = kwargs.get('file_meta_list', [])
        recursion = kwargs.get('recursion', 1)
        if len(file_meta_list)>=limit:
            return file_meta_list
        response_dict =  await self.ipfs_get_object(file_meta)
        response_links =  response_dict.get('Links')
        response_data = response_dict.get('Data', []) 
        folder_get_file_jobs = []
        if len(response_links)>0:
            job_list = []
            for response_link in response_links:
                
                job =  self.get_files(file_meta=response_link, file_meta_list=file_meta_list, recursion= recursion+1)
                job_list.append(job)
            await asyncio.gather(*job_list)
        elif len(response_links) == 0:

            file_meta_list  += response_data
    
        return file_meta_list
        # folder_get_file_jobs = []
        # for link_file_meta in response_links:
        #     job = await self.get_files(file_meta=link_file_meta, file_meta_list=file_meta_list)
        #     folder_get_file_jobs.append(job)

        # if folder_get_file_jobs:
        #     return await asyncio.gather(*folder_get_file_jobs)

        
    
    async def ipfs_cat(self, file_meta, timeout=10, action='post'):
        address = self.ipfs_url + '/cat'
        if action == 'get':
            response = await asyncio.wait_for(self.api_get( url=address,  params={'arg': file_meta['Hash']}), timeout=timeout)
        elif action == 'post':
            response = await asyncio.wait_for(self.api_post( url=address, params={'arg': file_meta['Hash']}), timeout=timeout)
        # except Exception as E:
        #     logger.error(f"Failed to get from IPFS {file_meta['Name']} {E}")
        #     return None
        return await response.json()

    async def ipfs_object_get(self, file_meta, timeout=2):
        response = await asyncio.wait_for(self.api_post( url=self.ipfs_url+'/object/get',  params={'arg': file_meta['Hash']}), timeout=timeout)
        return await response.json()


    async def ipfs_get_object(self, file_meta, timeout=1, action='post'):
        address = self.ipfs_url + '/object/get'
        if action == 'get':
            response = await asyncio.wait_for(self.api_get( url=address,  params={'arg': file_meta['Hash']}), timeout=timeout)
        elif action == 'post':
            response = await asyncio.wait_for(self.api_post( url=address, params={'arg': file_meta['Hash']}), timeout=timeout)
        # except Exception as E:
        #     logger.error(f"Failed to get from IPFS {file_meta['Name']} {E}")
        #     return None
        return await response.json()

    async def get_ipfs_directory(self, address: str, file_meta: dict, action: str = 'post', timeout : int = 5):
        r"""Connects to IPFS gateway and retrieves directory.
        Args:
            address: (:type:`str`, required):
                The target address of the request. 
            params: (:type:`tuple`, optional):
                The arguments of the request. eg. (('arg', dataset_hash),)
            action: (:type:`str`, optional):
                POST or GET.
            timeout: (:type:`int`, optional):
                Timeout for getting the server's response. 
        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        # session = requests.Session()
        # session.params.update((('arg', file_meta['Hash']), ))
        # try:
        if action == 'get':
            response = await asyncio.wait_for(self.api_get( url=address,  params={'arg': file_meta['Hash']}), timeout=timeout)
        elif action == 'post':
            response = await asyncio.wait_for(self.api_post( url=address, params={'arg': file_meta['Hash']}), timeout=timeout)
        # except Exception as E:
        #     logger.error(f"Failed to get from IPFS {file_meta['Name']} {E}")
        #     return None


        return await response.json()

    async def get_client_session(self, *args,**kwargs):
        # timeout = aiohttp.ClientTimeout(sock_connect=1, sock_read=5)
        # kwargs = {"timeout": timeout, **kwargs}
        return aiohttp.ClientSession(loop=self.loop, **kwargs)





    async def api_post(self, url, return_json = True, content_type=None, chunk_size=1024, num_chunks=None, **kwargs):
        
        headers = kwargs.pop('headers', {}) 
        params = kwargs.pop('params', kwargs)
        return_result = None
        async with aiohttp.ClientSession(loop=self.loop) as session:
            async with session.post(url,params=params,headers=headers) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                if num_chunks:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        st.write(data)
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    # async def api_get(self, url, session=None, **kwargs):
    #     if session == None:
    #         session = await self.get_client_session()
    #     headers = kwargs.pop('headers', {}) 
    #     params = kwargs.pop('params', kwargs)
    #     res = await session.get(url, params=params,headers=headers)
    #     await session.close()
    #     return res


    async def api_get(self, url, return_json = True, content_type=None, chunk_size=1024, num_chunks=1,**kwargs):
        
        headers = kwargs.pop('headers', {}) 
        params = kwargs.pop('params', kwargs)
        return_result = None
        async with aiohttp.ClientSession(loop=self.loop) as session:
            async with session.get(url,params=params,headers=headers) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                if chunk_size:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        st.write(data)
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result


    ##############
    #   ASYNCIO
    ##############
    @staticmethod
    def reset_event_loop(set_loop=True):
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop

    def set_event_loop(self, loop=None):
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop
    def get_event_loop(self):
        return asyncio.get_event_loop()

    def resolve_event_loop(self,loop=None):
        if loop == None:
            loop = self.new_event_loop()
        return loop
         
    def async_run(self, job, loop=None): 
        if loop == None:
            loop = self.loop
        return self.loop.run_until_complete(job)



Dataset()   


# import aiohttp
# import asyncio
# import time

# start_time = time.time()


# async def get_pokemon(session, url):
#     async with session.get(url) as resp:
#         pokemon = await resp.json()
#         return pokemon['name']


# async def main():

#     async with bittensor.dataset().get_client_session() as session:

#         tasks = []
#         for number in range(1, 151):
#             url = f'https://pokeapi.co/api/v2/pokemon/{number}'
#             tasks.append(asyncio.ensure_future(get_pokemon(session, url)))

#         original_pokemon = await asyncio.gather(*tasks)
#         for pokemon in original_pokemon:

# # # asyncio.run(main())
# # st.write("--- %s seconds ---" % (time.time() - start_time))
