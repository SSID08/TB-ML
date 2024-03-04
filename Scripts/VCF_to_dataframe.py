#import pickle
import subprocess as sp 
from collections import defaultdict
import numpy as np
import pandas as pd
#import argparse
import os
import re

'''
This script creates feature matrix from VCF files for downstream ML analysis
'''

lineage_dict={}

with open('./Data/lineage_table/lineage_tracker.txt','r') as f:
    for l in f:
        sample,lineage=l.strip().split(sep='\t')
        lineage_dict[sample]=lineage

#for file in ['pyrazinamide.vcf.gz','ofloxacin.vcf.gz','clarithromycin.vcf.gz']:
for file in os.listdir('./Files/Per_Drug_VCF_final'):
    if file.endswith('.gz'):
        genos=defaultdict(dict)
        f=os.path.join('./Files/Per_Drug_VCF_final',file)
        b_file=re.sub('.vcf.*','',file)
        print(b_file)
        pheno_file=os.path.join('./Sample_Phenotypes',f"{b_file}Sample_Phenotypes.txt")
        for l in sp.Popen(r"bcftools query -i '(TYPE=" + r'"snp" | TYPE="indel")' + r"'" + f" -f '[%POS\t%SAMPLE\t%GT\n]' {f}",shell=True,stdout=sp.PIPE).stdout:
            pos,sample,gt=l.decode().strip().split() # split each line on tab
            pos=int(pos)
            if gt=="./." or gt=="0/0":#infer genotype value from line
                gt = 0
            else:
                gt = 1

            try:
                #See if sample key exists in dictionary 
                genos[sample]
                try:
                    #See if 'pos' key exists for that particular sample
                    if genos[sample][pos]==0 and gt==1: # only if curent genotype from line is 1 and existing call in dictionary is zero
                        # add the genotype to the dictionary
                        genos[sample][pos] = gt
                except KeyError: # if 'pos' key not found in dictionary 
                    # add the genotype to the dictionary
                    genos[sample][pos] = gt 
            except KeyError: # if 'sample' key not found in dictionary
                genos[sample][pos] = gt             
                # add the genotype to the dictionary
    
        pheno={}

        with open(pheno_file,'r') as f: #Read phenotype file line by line
            next(f)
            for l in f:
                row=l.strip().split()
                pheno[row[0]]=int(row[1])
        try:
            my_keys=list(genos[list(genos.keys())[0]].keys())
            X=[]
            Y=[]
            lineage_list=[]
            for s in pheno:
                if s in genos:
                    X.append([genos[s][key] for key in my_keys])
                    Y.append(pheno[s])
                    if s in lineage_dict:
                        lineage_list.append(lineage_dict[s])
                    else:
                        lineage_list.append(np.nan)
                    #lineage_list.append(lineage_dict[s] if s in l)
            #X=[list(genos[s].values()) for s in pheno if s in genos]
            #samples=[s for s in pheno if s in genos]
            #Y = [pheno[s] for s in samples]            
            #lineage_list=[lineage_dict[s] if s in lineage_dict else np.nan for s in samples]

        except Exception as e:
            print(f'Exception: {e}')

        try:
            genos_matrix=np.array(X)
            pd_df=pd.DataFrame(data=genos_matrix,columns=my_keys)
            pd_df['Lineage']=np.array(lineage_list)
            pd_df['Phenotype']=np.array(Y)
            pd_df.to_pickle(path=f'./Updated_Data/Whole_Genome_dataframes/{b_file}.pkl')
        except Exception as e:
            print (f"File : {file} \t Error : {e}")