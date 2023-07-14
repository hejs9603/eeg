import mne
import pandas as pd
import os
from tqdm import tqdm

from mne.io import BaseRaw, RawArray
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, corrmap
import matplotlib.pyplot as plt

#%%
global raw

def read_vhdr_ica(origin_raw:BaseRaw):

    """
        MNE-Python supports a variety of preprocessing approaches and techniques
        (maxwell filtering, signal-space projection, independent components analysis,
        filtering, downsampling, etc); see the full list of capabilities in the mne.preprocessing
        and mne.filter submodules. Here we’ll clean up our data by performing independent components analysis (ICA);
        for brevity we’ll skip the steps that helped us determined which components best capture the artifacts
         (see Repairing artifacts with ICA for a detailed walk-through of that process).

    """
    raw1 = origin_raw.copy()

    #——————————————————————————————————————————————眼电通道垂直水平
    method = lambda data: data[0]-data[1]
    roi_dict = dict(
        Eog_v=mne.pick_channels(raw1.info["ch_names"], include=['VPVA','VNVB']),
        Eog_h=mne.pick_channels(raw1.info["ch_names"], include=['HPHL','HNHR']),
    )
    raw2 = mne.channels.combine_channels(raw1, groups=roi_dict, method=method)

    #——————————————————————————————————————————————抛弃原始眼电，肌电
    raw1.drop_channels(['VPVA', 'VNVB', 'HPHL', 'HNHR'])
    raw1.drop_channels(['OrbOcc', 'Mass'])  # 眼眶电 心电

    #——————————————————————————————————————————————添加眼电新通道
    raw1.add_channels([raw2])

    #——————————————————————————————————————————————心电通道修复
    # raw1.rename_channels( {'Erbs':'Ecg' } )

    raw1.rename_channels( {'Eog_v':'F9' } )
    raw1.rename_channels( {'Eog_h':'F10' } )
    raw1.rename_channels( {'Erbs':'Nz' } )

    # raw1.drop_channels(['Eog_h', 'Ecg'])


    #——————————————————————————————————————————————滤波 ==> 基线工频
    raw1.filter(l_freq=1, h_freq=30, fir_design="firwin", verbose=False) # 带通
    # raw1.plot()

    # 重参考
    # raw1.add_reference_channels(ref_channels='Fz')
    # chh = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    raw1.set_eeg_reference(ref_channels='average', verbose=False)

    # 定位
    raw1.set_montage('standard_1005', verbose=False, on_missing='ignore')

    raw1.rename_channels({'F9' : 'Eog_v'})
    raw1.rename_channels({'F10' : 'Eog_h'})
    raw1.rename_channels({'Nz' : 'Ecg'})

    return raw1
#%%

if __name__ == '__main__':
    # 疾病种类
    pclass = ['healthy']

    # 疾病对应的被试读取
    content = pd.read_excel("../TDBRAIN_participants_提取处理.xlsx", sheet_name=pclass)
    # ________________________________________________所有被试个体图像聚类列表_______________________________________________
    # raw_list = []
    # ica_list = []

    # 不同疾病循环
    for process_class in pclass:
        # 逐个读取各个疾病的静息态数据
        filenamelist = content[process_class]["participants_ID"]

        # subject level clustering
        pbar = tqdm(filenamelist[1:5])
        for j, f in enumerate(filenamelist[1:5]):
            pbar.update(1)

            for root, dirs, files in os.walk('H:/brain/实验-03/data/'+process_class+'/'+f+'/ses-1/eeg/'):
                for vhdrfile in files:
                    if "restEO_eeg.vhdr" in vhdrfile:
                        bids_fname = 'H:/brain/实验-03/data/'+process_class+'/'+f+'/ses-1/eeg/' + vhdrfile

                        # 文件读取，滤波(read_vhdr)
                        raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
                        raw = read_vhdr_ica(raw)

                        # 成分提取
                        ica = ICA(n_components=15, max_iter="auto", random_state=97)
                        ica.fit(raw)

                        # raw_list.append(raw)
                        # ica_list.append(ica)

                        # 使用EOG 去除眼电
                        ica.exclude = []
                        # find which ICs match the EOG pattern
                        eog_indices1, eog_scores1 = ica.find_bads_eog(raw, ch_name='Eog_v', verbose=False)
                        eog_indices2, eog_scores2 = ica.find_bads_eog(raw, ch_name='Eog_h', verbose=False)

                        # find which ICs match the ECG pattern
                        ecg_indices3, ecg_scores3 = ica.find_bads_ecg(raw, ch_name='Ecg', verbose=False)

                        ica.exclude.extend(eog_indices1)
                        ica.exclude.extend(eog_indices2)
                        ica.exclude.extend(ecg_indices3)

# # 以下为弃用代码，使用corrmap 计算组间成分的相似性，前提假设噪声是相似的
#     #%%
#     # use the first subject as template; use Fpz as proxy for EOG
#     raw = raw_list[0]
#     ica = ica_list[0]
#
#     # find which ICs match the EOG pattern
#     eog_indices1, eog_scores1 = ica.find_bads_eog(raw, ch_name='Eog_v', verbose=False)
#     eog_indices2, eog_scores2 = ica.find_bads_eog(raw, ch_name='Eog_h', verbose=False)
#
#     # find which ICs match the ECG pattern
#     ecg_indices3, ecg_scores3 = ica.find_bads_ecg(raw, ch_name='Ecg', verbose=False)
#
#     corrmap(ica_list, template=(0, eog_indices1[0]))
#
#
#     #%%
#     for index, (ica, raw) in enumerate(zip(ica_list, raw_list)):
#         with mne.viz.use_browser_backend("matplotlib"):
#             fig = ica.plot_sources(raw, show_scrollbars=False)
#         fig.subplots_adjust(top=0.9)  # make space for title
#         fig.suptitle("Subject {}".format(index))
#         plt.show()
#
#     corrmap(ica_list, template=(0, eog_indices1[0]), threshold=0.9, label="blink", plot=False)
#     print([ica.labels_ for ica in ica_list])
#
#     ica_list[3].plot_components(picks=ica_list[3].labels_["blink"])
#     ica_list[3].exclude = ica_list[3].labels_["blink"]
#     ica_list[3].plot_sources(raw_list[3], show_scrollbars=False)
#     plt.show()