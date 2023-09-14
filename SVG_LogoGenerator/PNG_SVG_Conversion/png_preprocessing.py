
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
from collections import defaultdict
from skimage.metrics import mean_squared_error
import glob
import os






class PNG_Preprocessor():
    
    def __init__(self):
        pass
    

    
    
    
    @staticmethod 
    def quantize_image (image,
                       error_threshold = 0.002,
                       show_intermediate_results = False):

        


        t = []
        for i in range(1,30):

            n_colors = i


            img = image



            by_color = defaultdict(int)
            for pixel in img.getdata():
                by_color[pixel] += 1





            # Convert to floats instead of the default 8 bits integer coding. Dividing by
            # 255 is important so that plt.imshow behaves works well on float data (need to
            # be in the range [0-1])
            img = np.array(img, dtype=np.float64) / 255
            

            # Load Image and transform to a 2D numpy array.
            w, h, d = original_shape = tuple(img.shape)
            assert d == 3
            image_array = np.reshape(img, (w * h, d))





            # Fitting model on a small sub-sample of the data
            image_array_sample = shuffle(image_array, random_state=0, n_samples= image_array.shape[0]//6)
            kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)

            
            # Get labels for all points
    
    
#           Predicting color indices on the full image (k-means)")
            labels = kmeans.predict(image_array)





            def recreate_image(codebook, labels, w, h):
                """Recreate the (compressed) image from the code book & labels"""
                return codebook[labels].reshape(w, h, -1)

            if show_intermediate_results:
                # Display all results, alongside original image
                plt.figure(1)
                plt.clf()
                plt.axis("off")
                plt.title(f"Original image {len(by_color)} colors)")
                plt.imshow(img)

                plt.figure(2)
                plt.clf()
                plt.axis("off")
                plt.title(f"Quantized image ({n_colors} colors, K-Means)")
                plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))





            mse = mean_squared_error(img,recreate_image(kmeans.cluster_centers_, labels, w, h))
            plt.show()



            t.append((i,round(mse,4)))

            if mse < error_threshold:
                img_array = recreate_image(kmeans.cluster_centers_, labels, w, h)
                img_quantized = Image.fromarray(((img_array * 255).astype(np.uint8)))

                break


        return img_quantized
    
    
    
    
        
        
        
        
        
        
        
        
        
        
    @staticmethod
    
    def quantize_image_from_path (
                                  input_image_path,
                                  write_to_file = False,
                                  output_folder = False,
                                  output_filename = False,
                                  error_threshold = 0.01, 
                                  show_intermediate_results = False):
        
        input_extension = "." + input_image_path.split("/")[-1].split(".")[-1]
        input_filename = input_image_path.split("/")[-1].split(".")[-2]
        input_folder = os.path.split(input_image_path)[0]
        
        
        image  = Image.open(input_image_path)
        
        outcome_image = PNG_Preprocessor.quantize_image(image = image,
                                       error_threshold = error_threshold,
                                       show_intermediate_results = show_intermediate_results)
        
        if write_to_file:
            
            if output_folder:
                
                if output_filename:
                    
                    outcome_image.save(output_folder+ "/"+output_filename+input_extension)
                    
                else:
                    
                    outcome_image.save(output_folder+ "/"+input_filename + "_quantized" + input_extension)
                    
            else:
                if output_filename:
                    outcome_image.save(input_folder+ "/" +output_filename+input_extension)
                    
                else:
                    outcome_image.save(input_folder+ "/" +input_filename + "_quantized" +  input_extension)
                    
                    
                    
            
            
        return outcome_image

    
    @staticmethod
    def increase_image_resolution (image,
                                   new_width ):
        
        img = image
        basewidth = new_width
#         img = Image.open(directory + \"/\"+ filename)\n",
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        
        
        return img
    
    @staticmethod
    def increase_image_resolution_path (
                                  input_image_path,
                                  new_width,
                                  write_to_file = False,
                                  output_folder = False,
                                  output_filename = False
                                        ):
        
        input_extension = "." + input_image_path.split("/")[-1].split(".")[-1]
        input_filename = input_image_path.split("/")[-1].split(".")[-2]
        input_folder = os.path.split(input_image_path)[0]
        
        
        image  = Image.open(input_image_path)
        
        
        outcome_image = PNG_Preprocessor.increase_image_resolution(
                                                                    image = image,
                                                                    new_width = new_width)
        
        
        if write_to_file:
            
            if output_folder:
                
                if output_filename:
                    
                    outcome_image.save(output_folder + "/" + output_filename+ input_extension)
                    
                else:
                    
                    outcome_image.save(output_folder + "/" +input_filename + f"_width_{new_width}" + input_extension)
                     
            else:
                if output_filename:
                    outcome_image.save(input_folder+ "/" + output_filename+input_extension)
                    
                else:
                    outcome_image.save(input_folder+ "/" + input_filename+ f"_width_{new_width}" +  input_extension)
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
