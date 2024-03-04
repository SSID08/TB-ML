import pickle
import argparse
import numpy as np
import os
import re
import subprocess as sp
from scipy.stats import fisher_exact,false_discovery_control
from scipy.stats.contingency import odds_ratio
import sys

parser=argparse.ArgumentParser(description='Run chi square tests on shortlisted stop gained/lost variants')

parser.add_argument('--input',help="Pickle file with matrix",required=True)
parser.add_argument('--out_folder',help="output folder path",required=True)
parser.add_argument('--anno_VCF',help='SNPeff annotated VCF',required=True)
args=parser.parse_args()

file=pickle.load(open(args.input,"rb"))
DR_name=re.sub('.pkl','',os.path.basename(args.input))
matrix=np.array(file['Matrix'])
gene_locii=np.array(file['Column_IDs'])
pheno=np.array(file['Phenotype'])
p_vals=[]
odds_ratio_list=[]
cont_matrices=[]
positive_phenos=np.where(pheno==1)[0]
negative_phenos=np.where(pheno==0)[0]
positive_genos_matrix=matrix[positive_phenos,:]
negative_genos_matrix=matrix[negative_phenos,:]
positive_genos_sums=np.sum(positive_genos_matrix,axis=0)
negative_genos_sums=np.sum(negative_genos_matrix,axis=0)

num_pos=len(positive_phenos)
num_neg=len(negative_phenos)
locii_inc=[]
for i in range(0,len(positive_genos_sums)):
    variant_in_resistant=positive_genos_sums[i]+1
    no_variant_in_resistant=num_pos-variant_in_resistant+2
    variant_in_susceptible=negative_genos_sums[i]+1
    no_variant_in_susceptible=num_neg-variant_in_susceptible+2
    if variant_in_resistant>5:
        try:
            input=np.array([[variant_in_resistant,no_variant_in_resistant],[variant_in_susceptible,no_variant_in_susceptible]])
            fisher_result=fisher_exact(input)
            odds_r=odds_ratio(input)
            odds_ratio_list.append(odds_r.statistic)
            p_vals.append(fisher_result.pvalue)
            cont_matrices.append(input)
            locii_inc.append(i)
        except Exception as e: 
            print(e)


p_adj=false_discovery_control(ps=np.array(p_vals),method='bh')
regions_dict={}
for i in range(0,len(p_adj)):
    if p_adj[i]<0.05:
        regions_dict[str(gene_locii[locii_inc[i]])]=[round(odds_ratio_list[i],3),"{:.3g}".format(p_adj[i]),cont_matrices[i]]

if not (len(odds_ratio_list)==len(locii_inc)==len(p_adj)):
    print('Error: Output sizes do not match')
    sys.exit(0)

out_dict={}

for l in sp.Popen(rf"bcftools query -f '%POS\t%AF\t%LOF\t%ANN\n' {args.anno_VCF}",shell=True,stdout=sp.PIPE).stdout:
    pos,AF,lof,ann=l.decode().strip().split()
    AF=float(AF)
    if lof!='.':
        try:
            pos=str(pos)
            OR=regions_dict[pos][0]
            p=regions_dict[pos][1]
            matrix=regions_dict[pos][2]
            ann=ann.split(',')[0].split('|')
            var_type,gene,subs,protein_subs=ann[1],ann[3],ann[9],ann[10]
            if pos in out_dict:
                if AF>out_dict[pos][0]:
                    out_dict[pos]=[AF,var_type,gene,subs,protein_subs,OR,p,lof,matrix]
            else:
                out_dict[pos]=[AF,var_type,gene,subs,protein_subs,OR,p,lof,matrix]
        except Exception as e:
            print(e)

with open(f"{args.out_folder}/{DR_name}_Associated_variants.txt",'w') as f: 
    f.write("Position\tOdds_Ratio\tMatrix\tP_value\tGene\tVariant_Type\tNucleotide_Subs\tProtein_Subs\tAllele Frequency\tLOF\n")
    for i in out_dict:
        try:
            arr=out_dict[i]
            f.write(f"{i}\t{arr[5]}\t{np.array2string(arr[8].flatten(),separator=',')}\t{arr[6]}\t{arr[2]}\t{arr[1]}\t{arr[3]}\t{arr[4]}\t{str(arr[0])}\t{arr[7]}\n")
        except Exception as e:
            print(f'Error {e} at position {i}')

print(f'Process done for drug {DR_name}')
