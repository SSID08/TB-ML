import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import re
import argparse
import subprocess as sp

parser=argparse.ArgumentParser(description='Plot multi-dimensional feature importances of model')

parser.add_argument('--model',help="Input model",required=True)
parser.add_argument('--df',help='Dataframe with genotypes and phenotypes',required=True)
parser.add_argument('--bed_file',help='Bed file for tracking gene names',required=True)
parser.add_argument('--label_threshold',help='%tile threshold for adding labels to plot',required=True)
parser.add_argument('--out_folder',help='Folder to store images',required=True)

args=parser.parse_args()
DR_name=re.sub('.pkl','',os.path.basename(args.df))

pickled_bed=pickle.load(open('../##path to bed file','rb'))
gene_ids,gene_names=pickled_bed[0],pickled_bed[1]

model=pickle.load(open(args.model,'rb'))['Model'].get_booster()
Info_gain=model.get_score(importance_type='total_gain')
Coverage=np.array(list(model.get_score(importance_type='total_cover').values()))/10**3
annotations=pickle.load(open(args.df,'rb'))['Columns']
non_zero_indices=[int(item[1:]) for item in Info_gain]
annotations=annotations[non_zero_indices]
Num_trees=np.array(list(model.get_score(importance_type='weight').values()))
Info_gain=np.log10(np.array(list(Info_gain.values())))
num_trees_threshold=np.percentile(Num_trees,q=int(args.label_threshold))
Info_gain_threshold=np.percentile(Info_gain,q=int(args.label_threshold))
for i in range(0,len(annotations)):
        out=re.sub('.*__','',annotations[i])
        if (Num_trees[i]>=num_trees_threshold and Info_gain[i]>=Info_gain_threshold):
                annotations[i]=out
        else:
                annotations[i]=''

var_postions=[f'Chromosome:{ann}' for ann in annotations if re.search('^[0-9]',ann)]
pos_to_ann={}
try: 
      for l in sp.Popen(rf"bcftools query -r {','.join(var_postions)} --regions-overlap 0 -f '%POS\t%TYPE\t%AC\t%ANN\n' ##VCF path"\
                        ,shell=True,stdout=sp.PIPE).stdout:
                 pos,type,AC,ann=l.decode().strip().split()
                 if type == 'SNP' or type =='INDEL':
                        AC=int(AC)
                        if str(pos) in annotations:
                              for i in ann.split(','):
                                    i_split=i.split('|')
                                    gene_name=i_split[3]
                                    gene_id=i_split[4]
                                    if (gene_name in gene_names) or (gene_id in gene_ids):
                                        gene_var=i_split[9]
                                        gene_var=f'{gene_name}_{gene_var}'
                                        if pos in pos_to_ann:
                                              if pos_to_ann[pos][1]>AC:
                                                    pos_to_ann[pos]=[gene_var,AC]   
                                        else: 
                                              pos_to_ann[pos]=[gene_var,AC]
                                        break
except Exception as e:
      print(f'Error {e} occured as exception')

out_labs= [pos_to_ann[ann][0] if re.search('^[0-9]',ann) else ann for ann in annotations] 

normalize = mcolors.Normalize(vmin=Coverage.min(), vmax=Coverage.max())
colormap = cm.YlOrRd

fig,ax=plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(8)
plt.scatter(x=Num_trees,y=Info_gain,c=colormap(normalize(Coverage)))
plt.axvline(x=num_trees_threshold,color='k',ls='--',ms=3)
plt.axhline(y=Info_gain_threshold,color='k',ls='--',ms=3)
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
for i in range(0,len(out_labs)):
        ax.annotate(out_labs[i],(Num_trees[i],Info_gain[i]),textcoords='offset points',xytext=(0,5),
                    ha='center',fontsize=10)
ax.set_xlabel('Feature Weight',fontsize=13)
ax.set_ylabel('Total Information Gain (log10)',fontsize=13)
sub_ax = plt.axes([0.87, 0.15, 0.03, 0.7])
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(Coverage)
cbar=plt.colorbar(scalarmappaple,cax=sub_ax)
cbar.set_label(label='Total Feature Coverage (scaled)',rotation=270,fontsize=12,labelpad=6)
plt.subplots_adjust(right=.85)
plt.savefig(f'{args.out_folder}/{DR_name}.png',dpi=500)
with open(f'{args.out_folder}/{DR_name}_HighScoringVariants.txt','w') as f: 
       for i in out_labs:
              if i!='':
                  f.write(f'{i}\n')
