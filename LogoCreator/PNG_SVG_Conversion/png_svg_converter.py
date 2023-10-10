import subprocess
import os


# vtracer is a command line app and usage pattern is:

# ./vtracer --input input.jpg --output output.svg #
# more about vtracer and usage available on github https://github.com/visioncortex/vtracer
        
class PNG_SVG_Converter ():
    
    def __init__(self, vtracer_path):
        self.vtracer_path = vtracer_path
        self.command = [self.vtracer_path, '--input',  '--output' ] 
        
    
    
    def convert_png_svg_path (self,
                              input_image_path,
                              output_folder = False,
                              output_filename = False):
        
        
        input_extension = "." + input_image_path.split("/")[-1].split(".")[-1]
        input_filename = input_image_path.split("/")[-1].split(".")[-2]
        input_folder = input_image_path[:-len(input_image_path.split("/")[-1])]
        
        
        
        file_command=[]
        
        ## in this order 
        
        file_command.append(self.command[0])
        file_command.append(self.command[1])
        file_command.insert(2, input_image_path)
        file_command.append(self.command[2])
        


        if output_folder:

            if output_filename:

                file_command.insert(4, output_folder + "/" + output_filename + ".svg" )
                
            else:
                
                file_command.insert(4, output_folder + "/" + input_filename + ".svg" )

        else:
            
            if output_filename:
                
                file_command.insert(4, input_folder + "/" + output_filename + ".svg" )
                
            else:
                
                file_command.insert(4, input_folder + "/" + output_filename + ".svg" )

        print(file_command)
        subprocess.run(file_command, capture_output = True)


        
        del file_command


        
    