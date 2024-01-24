""" Visualises SynTable generated annotations: """
# Run python ./visualize_annotations.py --dataset './sample_data' --ann_json './sample_data/annotation_final.json'
import json
import cv2
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from PIL import Image
import networkx as nx
import argparse

import pycocotools.mask as mask_util
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.patches as mpatches


# visualize annotations
def apply_mask(image, mask):
    # Convert to numpy arrays
    image = np.array(image)
    mask = np.array(mask)
    # Convert grayscale image to RGB
    mask = np.stack((mask,)*3, axis=-1)
    # Multiply arrays
    rgb_result= image*mask

    # First create the image with alpha channel
    rgba = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2RGBA)

    # Then assign the mask to the last channel of the image
    # rgba[:, :, 3] = alpha_data
    # Make image transparent white anywhere it is transparent
    rgba[rgba[...,-1]==0] = [255,255,255,0]

    return rgba

def compute_occluded_masks(mask1, mask2):
    """Computes occlusions between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    #if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        #return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    #masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    #masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    #area1 = np.sum(masks1, axis=0)
    #area2 = np.sum(masks2, axis=0)

    # intersections and union
    #intersections_mask = np.dot(masks1.T, masks2)
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection_mask = np.logical_and( mask1, mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection)

    return iou, intersection_mask.astype(float)

def convert_png(image):
    image = Image.fromarray(np.uint8(image))
    image = image.convert('RGBA')
    # Transparency
    newImage = []
    for item in image.getdata():
        if item[:3] == (0, 0, 0):
            newImage.append((0, 0, 0, 0))
        else:
            newImage.append(item)
    image.putdata(newImage)
    return image

def rle2mask(mask_rle, shape=(480,640)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

# Convert 1-channel groundtruth data to visualization image data
def normalize_greyscale_image(image_data):
    image_data = np.reciprocal(image_data)
    image_data[image_data == 0.0] = 1e-5
    image_data = np.clip(image_data, 0, 255)
    image_data -= np.min(image_data)
    if np.max(image_data) > 0:
        image_data /= np.max(image_data)
    image_data *= 255
    image_data = image_data.astype(np.uint8)
    return image_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise Annotations')
    parser.add_argument('--dataset', type=str,
                        help='dataset to visualise')
    parser.add_argument('--ann_json', type=str,
                        help='dataset annotation to visualise')
    args = parser.parse_args()
    data_dir = args.dataset
    ann_json = args.ann_json

    # Opening JSON file
    f = open(ann_json)
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()

    referenceDict = {}

    for i, ann in enumerate(data['annotations']):
        image_id = ann["image_id"]
        ann_id = ann["id"]
        # print(ann_id)
        if image_id not in referenceDict:
            referenceDict.update({image_id:{"rgb":None,"depth":None, "amodal":[], "visible":[],
    "occluded":[],"occluded_rate":[],"category_id":[],"object_name":[]}})
            # print(referenceDict)
            referenceDict[image_id].update({"rgb":data["images"][i]["file_name"]})
            referenceDict[image_id].update({"depth":data["images"][i]["depth_file_name"]})
            # referenceDict[image_id].update({"occlusion_order":data["images"][i]["occlusion_order_file_name"]})
            referenceDict[image_id]["amodal"].append(ann["segmentation"])
            referenceDict[image_id]["visible"].append(ann["visible_mask"])
            referenceDict[image_id]["occluded"].append(ann["occluded_mask"])
            referenceDict[image_id]["occluded_rate"].append(ann["occluded_rate"])
            referenceDict[image_id]["category_id"].append(ann["category_id"])
            # referenceDict[image_id]["object_name"].append(ann["object_name"])

        else:
            # if not (referenceDict[image_id]["rgb"] or referenceDict[image_id]["depth"]):
            #     referenceDict[image_id].update({"rgb":data["images"][i]["file_name"]})
            #     referenceDict[image_id].update({"depth":data["images"][i]["depth_file_name"]})
            referenceDict[image_id]["amodal"].append(ann["segmentation"])
            referenceDict[image_id]["visible"].append(ann["visible_mask"])
            referenceDict[image_id]["occluded"].append(ann["occluded_mask"])
            referenceDict[image_id]["occluded_rate"].append(ann["occluded_rate"])
            referenceDict[image_id]["category_id"].append(ann["category_id"])
            # referenceDict[image_id]["object_name"].append(ann["object_name"])

    # Create visualise directory
    vis_dir = os.path.join(data_dir,"visualise_dataset")
    if os.path.exists(vis_dir): # remove contents if exist
        for filename in os.listdir(vis_dir):
            file_path = os.path.join(vis_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(vis_dir)

    # query_img_id_list = [1,50,100]
    query_img_id_list = [i for i in range(1,len(referenceDict)+1)] # visualise all images
    for id in query_img_id_list:
        if id in referenceDict:
            ann_dic = referenceDict[id]
            vis_dir_img = os.path.join(vis_dir,str(id))
            if not os.path.exists(vis_dir_img):
                os.makedirs(vis_dir_img)

            # visualise rgb image
            rgb_path = os.path.join(data_dir,ann_dic["rgb"])
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

            # visualise depth image
            depth_path = os.path.join(data_dir,ann_dic["depth"])
            from PIL import Image
            im = Image.open(depth_path)
            im = np.array(im)
            depth_img = Image.fromarray(normalize_greyscale_image(im.astype("float32")))
            file = os.path.join(vis_dir_img,f"depth_{id}.png")
            depth_img.save(file, "PNG")

            # visualise occlusion masks on rgb image
            occ_img_list = ann_dic["occluded"]
            if len(occ_img_list) > 0:
                occ_img = rgb_img.copy()
                overlay = rgb_img.copy()
                combined_mask = np.zeros((occ_img.shape[0],occ_img.shape[1]))
                # iterate through all occlusion masks
                for i, occMask in enumerate(occ_img_list):
                    occluded_mask = mask_util.decode(occMask)
                    if ann_dic["category_id"][i] == 0:
                        occ_img_back = rgb_img.copy()
                        overlay_back = rgb_img.copy()
                        occluded_mask = occluded_mask.astype(bool) # boolean mask
                        overlay_back[occluded_mask] = [0, 0, 255]
                        # print(np.unique(occluded_mask))
                        alpha =0.5                  
                        occ_img_back = cv2.addWeighted(overlay_back, alpha, occ_img_back, 1 - alpha, 0, occ_img_back)      

                        occ_save_path = f"{vis_dir_img}/rgb_occlusion_{id}_background.png"
                        cv2.imwrite(occ_save_path, occ_img_back)
                    else:
                        combined_mask += occluded_mask

                combined_mask = combined_mask.astype(bool) # boolean mask
                overlay[combined_mask] = [0, 0, 255]
                
                alpha =0.5                  
                occ_img = cv2.addWeighted(overlay, alpha, occ_img, 1 - alpha, 0, occ_img)      

                occ_save_path = f"{vis_dir_img}/rgb_occlusion_{id}.png"
                cv2.imwrite(occ_save_path, occ_img)

                combined_mask = combined_mask.astype('uint8')
                occ_save_path = f"{vis_dir_img}/occlusion_mask_{id}.png"
                cv2.imwrite(occ_save_path, combined_mask*255)

                cols = 4
                rows = len(occ_img_list) // cols + 1
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, occMask in enumerate(occ_img_list):
                    occ_mask = mask_util.decode(occMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    # plt.title(ann_dic["object_name"][index])
                    plt.imshow(occ_mask)

                plt.tight_layout()
                plt.suptitle(f"Occlusion Masks for {id}.png")
                # plt.show()        
                plt.savefig(f'{vis_dir_img}/occ_masks_{id}.png')
                plt.close()
            
            #  visualise visible masks on rgb image
            vis_img_list = ann_dic["visible"]
            if len(vis_img_list) > 0:
                vis_img = rgb_img.copy()
                overlay = rgb_img.copy()
                # iterate through all occlusion masks
                for i, visMask in enumerate(vis_img_list):
                    visible_mask =  mask_util.decode(visMask)
                    
                    if ann_dic["category_id"][i] == 0:
                        vis_img_back = rgb_img.copy()
                        overlay_back = rgb_img.copy()
                        visible_mask = visible_mask.astype(bool) # boolean mask
                        overlay_back[visible_mask] = [0, 0, 255]
                        
                        alpha =0.5                  
                        vis_img_back = cv2.addWeighted(overlay_back, alpha, vis_img_back, 1 - alpha, 0, vis_img_back)      

                        vis_save_path = f"{vis_dir_img}/rgb_visible_mask_{id}_background.png"
                        cv2.imwrite(vis_save_path, vis_img_back)
                    else:
                        vis_combined_mask = visible_mask.astype(bool) # boolean mask      
                        colour = list(np.random.choice(range(256), size=3))
                        overlay[vis_combined_mask] = colour

                alpha = 0.5   
                vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)        
                vis_save_path = f"{vis_dir_img}/rgb_visible_mask_{id}.png"
                cv2.imwrite(vis_save_path,vis_img)

                cols = 4
                rows = len(vis_img_list) // cols + 1
                # print(len(amodal_img_list))
                # print(cols,rows)
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, visMask in enumerate(vis_img_list):
                    vis_mask = mask_util.decode(visMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    # plt.title(ann_dic["object_name"][index])
                    plt.imshow(vis_mask)

                plt.tight_layout()
                plt.suptitle(f"Visible Masks for {id}.png")
                # plt.show()        
                plt.savefig(f'{vis_dir_img}/vis_masks_{id}.png')
                plt.close()

            # visualise amodal masks
            # img_dir_path = f"{output_dir}/visualize_occlusion_masks/"
            # img_list = sorted(os.listdir(img_dir_path), key=lambda x: float(x[4:-4]))
            amodal_img_list = ann_dic["amodal"]
            if len(amodal_img_list) > 0:
                cols = 4
                rows = len(amodal_img_list) // cols + 1
                # print(len(amodal_img_list))
                # print(cols,rows)
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, amoMask in enumerate(amodal_img_list):
                    amodal_mask = mask_util.decode(amoMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    # plt.title(ann_dic["object_name"][index])
                    plt.imshow(amodal_mask)

                plt.tight_layout()
                plt.suptitle(f"Amodal Masks for {id}.png")
                # plt.show()        
                plt.savefig(f'{vis_dir_img}/amodal_masks_{id}.png')
                plt.close()
            
            # get rgb_path
            rgb_path = os.path.join(data_dir,ann_dic["rgb"])
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            occ_order = False
            if occ_order:
                # get occlusion order adjacency matrix
                npy_path = os.path.join(data_dir,ann_dic["occlusion_order"])
                occlusion_order_adjacency_matrix = np.load(npy_path)

                print(f"Calculating Directed Graph for Scene:{id}")
                # vis_img = cv2.imread(f"{vis_dir}/visuals/{scene_index}.png", cv2.IMREAD_UNCHANGED)
                rows = cols = len(ann_dic["visible"]) # number of objects
                obj_rgb_mask_list = []
                for i in range(1,len(ann_dic["visible"])+1):
                    visMask = ann_dic["visible"][i-1]
                    visible_mask = mask_util.decode(visMask)
                    
                    rgb_crop = apply_mask(rgb_img, visible_mask)
                    rgb_crop = convert_png(rgb_crop)
                    
                    def bbox(im):
                        a = np.array(im)[:,:,:3]  # keep RGB only
                        m = np.any(a != [0,0,0], axis=2)
                        coords = np.argwhere(m)
                        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
                        return (x0, y0, x1+1, y1+1)

                    # print(bbox(rgb_crop))
                    obj_rgb_mask = rgb_crop.crop(bbox(rgb_crop))

                    obj_rgb_mask_list.append(obj_rgb_mask) # add obj_rgb_mask

                    # get contours (presumably just one around the nonzero pixels)  # for instance segmentation mask
                    # contours = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # contours = contours[0] if len(contours) == 2 else contours[1]
                    # for cntr in contours:
                    #     x,y,w,h = cv2.boundingRect(cntr)
                    # cv2.putText(img=vis_img, text=str(i), org=(x+w//2, y+h//2), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)


                """ === Generate Directed Graph === """
                # print("Occlusion Order Adjacency Matrix:\n",occlusion_order_adjacency_matrix)
                # f, (ax1,ax2) = plt.subplots(1,2)
                # show_graph_with_labels(overlap_adjacency_matrix,ax1)
                labels = [i for i in range(1,len(occlusion_order_adjacency_matrix)+1)]
                labels_dict = {}
                for i in range(len(occlusion_order_adjacency_matrix)):
                    labels_dict.update({i:labels[i]})
                
                rows, cols = np.where(occlusion_order_adjacency_matrix == 1)
                rows += 1
                cols += 1
                edges = zip(rows.tolist(), cols.tolist())
                nodes_list = [i for i in range(1, len(occlusion_order_adjacency_matrix)+1)]
                # Initialise directed graph G
                G = nx.DiGraph()
                G.add_nodes_from(nodes_list)
                G.add_edges_from(edges)
                

                # pos=nx.spring_layout(G,k=1/sqrt(N))
                is_planar, P = nx.check_planarity(G)
                if is_planar:
                    pos=nx.planar_layout(G)
                else:
                    # pos=nx.draw(G)
                    N = len(G.nodes())
                    pos=nx.spring_layout(G,k=3/sqrt(N))

                print("Nodes:",G.nodes())
                print("Edges:",G.edges())
                # print(G.in_edges())
                # print(G.out_edges())
                # get start nodes
                start_nodes = [node for (node,degree) in G.in_degree if degree == 0]
                print("start_nodes:",start_nodes)
                # get end nodes
                end_nodes = [node for (node,degree) in G.out_degree if degree == 0]
                for node in end_nodes:
                    if node in start_nodes:
                        end_nodes.remove(node)
                print("end_nodes:",end_nodes)
                # get intermediate notes
                intermediate_nodes = [i for i in nodes_list if i not in (start_nodes) and i not in (end_nodes)]
                print("intermediate_nodes:",intermediate_nodes)

                print("(Degree of clustering) Number of Weakly Connected Components:",nx.number_weakly_connected_components(G))
                # largest_wcc = max(nx.weakly_connected_components(G), key=len)
                # largest_wcc_size = len(largest_wcc)
                # print("(Scene Complexity) Sizes of Weakly Connected Component:",largest_wcc_size)
                
                wcc_list = list(nx.weakly_connected_components(G))
                wcc_len = []
                for component in wcc_list:
                    wcc_len.append(len(component))
                print("(Scene Complexity/Degree of overlapping regions) Sizes of Weakly Connected Components:",wcc_len)

                dag_longest_path_length = nx.dag_longest_path_length(G)
                print("(Minimum no. of depth layers to order all regions in WCC) Longest directed path of Weakly Connected Components:",dag_longest_path_length)

                # nx.draw(gr, node_size=500, with_labels=True)
                node_color_list = []
                node_size_list = []
                for node in nodes_list:
                    if node in start_nodes:
                        node_color_list.append('green')
                        node_size_list.append(500)
                    elif node in end_nodes:
                        node_color_list.append('yellow')
                        node_size_list.append(300)
                    else:
                        node_color_list.append('#1f78b4')
                        node_size_list.append(300)

                options = {
                'node_color': node_color_list,
                'node_size': node_size_list,
                'width': 1,
                'arrowstyle': '-|>',
                'arrowsize': 10
                }
                fig1 = plt.figure(figsize=(20, 6), dpi=80)
                
                plt.subplot(1,3,1)
                # nx.draw_planar(G, pos, with_labels = True, arrows=True, **options)
                nx.draw_networkx(G,pos, with_labels= True, arrows=True, **options)

                dag = nx.is_directed_acyclic_graph(G)
                print(f"Is Directed Acyclic Graph (DAG)?: {dag}")
                

                colors = ["green", "#1f78b4", "yellow"]
                texts = ["Top Layer", "Intermediate Layers", "Bottom Layer"]
                patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                            label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
                plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), 
                        loc='center', ncol=3, fancybox=True, shadow=True, 
                        facecolor="w", numpoints=1, fontsize=10)
                plt.title("Directed Occlusion Order Graph")
                        
                # plt.subplot(1,2,2)
                # plt.imshow(vis_img)       
                # plt.imshow(vis_img)
                
                # plt.title(f"Visible Masks Scene {scene_index}")
                plt.axis('off')
                # plt.show()
                # plt.savefig(f"{output_dir}/vis_img_{i}.png")
                # cv2.imwrite(f"{output_dir}/scene_{scene_index}.png", vis_img)
                # plt.show()

                # fig2 = plt.figure(figsize=(16, 6), dpi=80)
                plt.subplot(1,3,2)
                options = {
                'node_color': "white",
                # 'node_size': node_size_list,
                'width': 1,
                'arrowstyle': '-|>',
                'arrowsize': 10
                }
                # nx.draw_networkx(G, arrows=True, **options)
                # nx.draw(G,  with_labels = True,arrows=True, connectionstyle='arc3, rad = 0.1')
                # nx.draw_spring(G,  with_labels = True,arrows=True, connectionstyle='arc3, rad = 0.5')
                
                N = len(G.nodes())
                from math import sqrt
                if is_planar:
                    pos=nx.planar_layout(G)
                else:
                    # pos=nx.draw(G)
                    N = len(G.nodes())
                    pos=nx.spring_layout(G,k=3/sqrt(N))

                nx.draw_networkx(G,pos, with_labels= False, arrows=True, **options)
                plt.title("Visualisation of Occlusion Order Graph")
                # draw with images on nodes
                # nx.draw_networkx(G,pos,width=3,edge_color="r",alpha=0.6)
                ax=plt.gca()
                fig=plt.gcf()
                trans = ax.transData.transform
                trans2 = fig.transFigure.inverted().transform
                imsize = 0.05 # this is the image size

                node_size_list = []
                for n in G.nodes():
                    (x,y) = pos[n]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    # a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
                    a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
                    a.imshow(obj_rgb_mask_list[n-1])
                    a.set_aspect('equal')
                    a.axis('off')
                # fig.patch.set_visible(False)
                ax.axis('off')

                plt.subplot(1,3,3)
                plt.imshow(rgb_img)       
                plt.axis('off')
                plt.title(f"RGB Scene {id}")
                # plt.tight_layout()
                # plt.show()
                plt.savefig(f'{vis_dir_img}/occlusion_order_{id}.png')
                plt.close()


                m = occlusion_order_adjacency_matrix.astype(int)
                unique_chars, matrix = np.unique(m, return_inverse=True)
                color_dict = {1: 'darkred', 0: 'white'}
                plt.figure(figsize=(20,20))
                sns.set(font_scale=2)
                ax1 = sns.heatmap(matrix.reshape(m.shape), annot=m, annot_kws={'fontsize': 20}, fmt='',
                                linecolor='dodgerblue', lw=5, square=True, clip_on=False,
                                cmap=ListedColormap([color_dict[char] for char in unique_chars]),
                                xticklabels=np.arange(m.shape[1]) + 1, yticklabels=np.arange(m.shape[0]) + 1, cbar=False)
                ax1.tick_params(labelrotation=0)
                ax1.tick_params(axis='both', which='major', labelsize=20, labelbottom = False, bottom=False, top = False, labeltop=True)
                plt.xlabel("Occludee")
                ax1.xaxis.set_ticks_position('top')
                ax1.xaxis.set_label_position('top')
                plt.ylabel("Occluder")
                # plt.show()
                plt.savefig(f'{vis_dir_img}/occlusion_order_adjacency_matrix_{id}.png')
                plt.close()
