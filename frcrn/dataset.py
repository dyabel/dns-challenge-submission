from modelscope.msdatasets import MsDataset

MsDataset.load(
        'ICASSP_2021_DNS_Challenge',
        namespace='modelscope',
        split='test')