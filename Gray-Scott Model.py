'''
Overview:
    --> Program creates four videos showing the results of various changes in the F and
        k parameters in the evolution of four of the shown 12 patterns from the Pearson paper
        for the Gray-Scott Model using 400 frame videos
        
    --> class Createvideo:
        --> Used exact Createvideo class given to us from Project_2_Video_Creator_Final.py
            with minor changes
        
    
Four Videos of Patterns:
    1.) F = 0.056 and k = 0.065 (mu Pearson paper pattern)
    2.) F = 0.03 and k = 0.055 (delta Pearson paper pattern)
    3.) F = 0.04 and k = 0.063 (kappa Pearson paper pattern)
    4.) F = 0.041 and k = 0.06 (theta Pearson paper pattern)


laplacian(array) function:
    --> Parameters:
        1.) input a 2D array
    
    --> Function used to calculate expression (Du*dell**2 * U) in reaction-diffusion 
        equation using cyclic boundary conditions
        
        
diffusion_eqns function:
    --> Parameters:
        1.) U array (U)
        2.) V array (V)
        3.) Diffusion coefficient for U (Du)
        4.) Diffusion coefficient for V (Dv)
        5.) Feed rate (F) paramter
        6.) Kill rate (k) parameter
        7.) Time step (dt)
    
    --> Function calculates values of U and V arrays based off of given PDEs


FuncAnimation Parameters:
    1) "plt.gcf()" = Get Current Figure
    2) "interval" = # Milliseconds Until Frame is Updated
    3) "frames" = # of Total Frames to Animate
    4) "repeat" = Repeat Animation Video [True/False]
    5) "blit" = Only Redraw Pieces that Changes [True/False]

Calculate Total Time for Video (If # of Total Frames is Known): [Equations]
    -> (Frames) * (Interval / 1000) = # Seconds
    - (1000 / Interval) = FPS = Frames Per Second

'''


# imported modules
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# code for video-creating class from Project_2_Video_Creator_Final.py
class CreateVideo:
    def __init__(self, array, title, save, frames=200, interval=20, repeat=False, blit=True, show: bool = True):
        """
        Get Values to Generate Video Class
        :param array: Total Points to Animate (Array)
        :param title: Graph Title
        :param save: Title of Saved Video
        :param frames: # of Total Frames to Animate
        :param interval: # Milliseconds Until Frame is Updated
        :param repeat: Repeat Animation Video [True/False]
        :param blit: Only Redraw Pieces that Changes [True/False]
        :param show: Show the Animation in Matplotlib [True/False]
        """
        self.array = array
        self.title = title
        self.save = save
        self.frames = frames
        self.interval = interval
        self.repeat = repeat
        self.blit = blit
        self.show = show
        self.image = None
        self.setup()

    # Animate Function
    def animate(self, i):
        """
        Animation Function that will Run for Each Frame
        :param i: Each Frame Iteration
        """
        # Plot Values
        self.image.set_array(self.array[i, :, :])  # Change to self.array[i, :, :]
        return [self.image]

    # Setup Function
    def setup(self):
        # Figure Settings
        fig, ax = plt.subplots()
        self.image = ax.imshow(self.array[0, :, :], interpolation="bicubic", cmap="jet")  # Change to self.array[i, :, :]
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        fig.colorbar(self.image, ax=ax)
        ax.set_title(self.title)
        fig.tight_layout()

        # Generate Animation
        output = animation.FuncAnimation(plt.gcf(), self.animate, frames=self.frames, interval=self.interval,
                                         repeat=self.repeat, blit=self.blit)

        # Save Output
        print(">>> Generating Video")
        output.save(f"{self.save}.mp4", writer='ffmpeg', fps=30)
        print(">>> Video Generated")

        # Show Plot
        if self.show:
            plt.show()




# Parameters:

# Diffusion coeffeicent for u
Du = 2e-5
# Diffusion coefficent for v
Dv = 1e-5
# Dimensionless feed rates (F)
F_values = [0.056, 0.03, 0.04, 0.041]
# Dimensionless Kill Rates (k)
k_values = [0.065, 0.055, 0.063, 0.06]
# time step
dt = 1
# number of frames
frames = 400
# Grid length
N = 256

# h-value used to scale laplacian value in laplacian function (while using Du = 2e-5 and Dv = 1e-5)
h = 2.5 / N

# Initializing all 2D-array values of U to 1
U = np.ones((N+1, N+1))
# Initializing all 2D-array values of V to 0
V = np.zeros((N+1, N+1))


# scaling factor of length divided by 10 that will be used to calculate middle section of array
r = N // 10
# start index for calculating middle section of array
startIndex = N//2 - r//2

# Sets center of grid for U equal to 0.5
U[startIndex:startIndex + r, startIndex:startIndex + r] = 0.5
# Sets center of grid for V equal to 0.25
V[startIndex:startIndex + r, startIndex:startIndex + r] = 0.25

# Creates random array with values between -0.1 and 0.1 in an arary of size N+1 by N+1
perturbations = np.random.random((N+1, N+1))

# Multiplies all of these values by 0.1 to caculate the 1% random noise needed to break the square symmetry
perturbations = perturbations * 0.1

# Adding the peturbations needed to break apart U square symmetry
U = U + perturbations

# Adding the peturbations needed to break apart V square symmetry
V = V + perturbations


# Used to calculate expression (Du*dell**2 * U) in reaction-diffusion equation using cyclic boundary conditions
def laplacian(array):
    # computes laplacian of a 2D array with periodic boundary conditions
    laplacian = (np.roll(array, -1, axis = 0) + np.roll(array, 1, axis = 0) +
                 np.roll(array, -1, axis = 1) + np.roll(array, 1, axis = 1) 
                 -4 * array
                )
    
    # divides laplacian by scaling factor of h**2
        # ensures that when using the specified Du and Dv values laplacian function outputs proper value
    return laplacian / (h)**2



# Calculates values of U and V arrays based off of given PDEs
def diffusion_eqns(U, V, Du, Dv, F, k, dt):
    # Calls laplacian function for both U and V and sets equal to variable
    U_laplacian = laplacian(U)
    V_laplacian = laplacian(V)
    
    # Calculates value of U with specified parameters
    U_updated = U + dt * (Du * U_laplacian - U * V**2 + F * (1-U))
    
    # Calculates value of V with specified parameters
    V_updated = V + dt * (Dv * V_laplacian + U * V**2 - (F + k) * V)
    
    # Returns updated U and V values for the given parameters
    return U_updated, V_updated



# sets various parameter values for CreateVideo class instance calls
ylim = None
frames = 400
interval = 100  # Equals the time spent on each frame
repeat = False
blit = True
show = False
totalTime = 200000




'''
## Code for creating video for U Field Evolution for F = 0.056 and k = 0.065 ##
## Creates video correlating to pattern labeled by mu variable in Pearson paper ##
'''

# 3D array initialized to hold 400 frames of U array as it is updated
U_array = np.zeros((frames, N+1, N+1))
# 3D array initialized to hold 400 frames of V array as it is updated
V_array = np.zeros((frames, N+1, N+1))

# Initializes F = 0.056
F = F_values[0]
# Initializes k = 0.065
k = k_values[0]

# Step size
# Value of 500 used to have 400 frames saved because frame = totalTime / step
step = 500

# for-loop iterates for 200000 time steps and updates U and V arrays at each time step
# saves 
for i in range(totalTime):
    # Update the U and V arrays
    U, V = diffusion_eqns(U, V, Du, Dv, F, k, dt)
    
    # Plots and saves images at 500 step so that 400 images will be saved for use in creating the video
    if i % step == 0:
        # Sets updated value into proper place in 3D array of U that holds 400 frames
        U_array[i//step] = U
        # Sets updated value into proper place in 3D array of V that holds 400 frames
        V_array[i//step] = V
        
        # This code properly plots and saves the images from the U array
        # Do not need to save the image for V because we only need to plot U for this project
        plt.imshow(U, interpolation='bicubic', cmap=plt.cm.jet)
        plt.savefig("imshow-{:03d}.png".format(i))
        plt.close()


# Sets title and save name for video
title = "Gray-Scott Model for U with F = 0.056 and k = 0.065"
saveName = "Gray-Scott Model (F=0.056, K=0.065)"
# Creates video from the 400 saved images/frames
output = CreateVideo(U_array, title, saveName, frames, interval, repeat, blit, show)




'''
## Code for creating video for U Field Evolution for F = 0.03 and k = 0.055 ##
## Creates video correlating to pattern labeled by delta variable in Pearson paper ##
'''
    
# Initializing all 2D-array values of U to 1
U = np.ones((N+1, N+1))
# Initializing all 2D-array values of V to 0
V = np.zeros((N+1, N+1))

# Sets center of grid for U equal to 0.5
U[startIndex:startIndex + r, startIndex:startIndex + r] = 0.5
# Sets center of grid for V equal to 0.25
V[startIndex:startIndex + r, startIndex:startIndex + r] = 0.25

# Creates random array with values between -0.1 and 0.1 in an arary of size N+1 by N+1
perturbations = np.random.random((N+1, N+1))

# Multiplies all of these values by 0.1 to caculate the 1% random noise needed to break the square symmetry
perturbations = perturbations * 0.1

# Adding the peturbations needed to break apart U square symmetry
U = U + perturbations

# Adding the peturbations needed to break apart V square symmetry
V = V + perturbations


# 3D array initialized to hold 400 frames of U array as it is updated
U_array = np.zeros((frames, N+1, N+1))
# 3D array initialized to hold 400 frames of V array as it is updated
V_array = np.zeros((frames, N+1, N+1))


# Initializes F = 0.03
F = F_values[1]
# Initializes k = 0.055
k = k_values[1]

# for-loop iterates for 200000 time steps and updates U and V arrays at each time step
# saves 
for i in range(totalTime):
    # Update the U and V arrays
    U, V = diffusion_eqns(U, V, Du, Dv, F, k, dt)
    
    # Plots and saves images at 500 step so that 400 images will be saved for use in creating the video
    if i % step == 0:
        # Sets updated value into proper place in 3D array of U that holds 400 frames
        U_array[i//step] = U
        # Sets updated value into proper place in 3D array of V that holds 400 frames
        V_array[i//step] = V
        
        # This code properly plots and saves the images from the U array
        # Do not need to save the image for V because we only need to plot U for this project
        plt.imshow(U, interpolation='bicubic', cmap=plt.cm.jet)
        plt.savefig("imshow-{:03d}.png".format(i))
        plt.close()
        

# Sets title and save name for video
title = "Gray-Scott Model for U with F = 0.03 and k = 0.055"
saveName = "Gray-Scott Model (F=0.03, K=0.055)"
# Creates video from the 400 saved images/frames
output = CreateVideo(U_array, title, saveName, frames, interval, repeat, blit, show)




'''
## Code for creating video for U Field Evolution for F = 0.04 and k = 0.063 ##
## Creates video correlating to pattern labeled by kappa variable in Pearson paper ##
'''
    
# Initializing all 2D-array values of U to 1
U = np.ones((N+1, N+1))
# Initializing all 2D-array values of V to 0
V = np.zeros((N+1, N+1))

# Sets center of grid for U equal to 0.5
U[startIndex:startIndex + r, startIndex:startIndex + r] = 0.5
# Sets center of grid for V equal to 0.25
V[startIndex:startIndex + r, startIndex:startIndex + r] = 0.25

# Creates random array with values between -0.1 and 0.1 in an arary of size N+1 by N+1
perturbations = np.random.random((N+1, N+1))

# Multiplies all of these values by 0.1 to caculate the 1% random noise needed to break the square symmetry
perturbations = perturbations * 0.1

# Adding the peturbations needed to break apart U square symmetry
U = U + perturbations

# Adding the peturbations needed to break apart V square symmetry
V = V + perturbations


# 3D array initialized to hold 400 frames of U array as it is updated
U_array = np.zeros((frames, N+1, N+1))
# 3D array initialized to hold 400 frames of V array as it is updated
V_array = np.zeros((frames, N+1, N+1))


# Initializes F = 0.04
F = F_values[2]
# Initializes k = 0.063
k = k_values[2]


# for-loop iterates for 200000 time steps and updates U and V arrays at each time step
# saves 
for i in range(totalTime):
    # Update the U and V arrays
    U, V = diffusion_eqns(U, V, Du, Dv, F, k, dt)
    
    # Plots and saves images at 500 step so that 400 images will be saved for use in creating the video
    if i % step == 0:
        # Sets updated value into proper place in 3D array of U that holds 400 frames
        U_array[i//step] = U
        # Sets updated value into proper place in 3D array of V that holds 400 frames
        V_array[i//step] = V
        
        # This code properly plots and saves the images from the U array
        # Do not need to save the image for V because we only need to plot U for this project
        plt.imshow(U, interpolation='bicubic', cmap=plt.cm.jet)
        plt.savefig("imshow-{:03d}.png".format(i))
        plt.close()


# Sets title and save name for video
title = "Gray-Scott Model for U with F = 0.04 and k = 0.063"
saveName = "Gray-Scott Model (F=0.04, K=0.063)"
# Creates video from the 400 saved images/frames
output = CreateVideo(U_array, title, saveName, frames, interval, repeat, blit, show)



'''
## Code for creating video for U Field Evolution for F = 0.041 and k = 0.06 ##
## Creates video correlating to pattern labeled by theta variable in Pearson paper ##
'''
    
# Initializing all 2D-array values of U to 1
U = np.ones((N+1, N+1))
# Initializing all 2D-array values of V to 0
V = np.zeros((N+1, N+1))

# Sets center of grid for U equal to 0.5
U[startIndex:startIndex + r, startIndex:startIndex + r] = 0.5
# Sets center of grid for V equal to 0.25
V[startIndex:startIndex + r, startIndex:startIndex + r] = 0.25

# Creates random array with values between -0.1 and 0.1 in an arary of size N+1 by N+1
perturbations = np.random.random((N+1, N+1))

# Multiplies all of these values by 0.1 to caculate the 1% random noise needed to break the square symmetry
perturbations = perturbations * 0.1

# Adding the peturbations needed to break apart U square symmetry
U = U + perturbations

# Adding the peturbations needed to break apart V square symmetry
V = V + perturbations


# 3D array initialized to hold 400 frames of U array as it is updated
U_array = np.zeros((frames, N+1, N+1))
# 3D array initialized to hold 400 frames of V array as it is updated
V_array = np.zeros((frames, N+1, N+1))


# Initializes F = 0.041
F = F_values[3]
# Initializes k = 0.06
k = k_values[3]


# for-loop iterates for 200000 time steps and updates U and V arrays at each time step
# saves 
for i in range(totalTime):
    # Update the U and V arrays
    U, V = diffusion_eqns(U, V, Du, Dv, F, k, dt)
    
    # Plots and saves images at 500 step so that 400 images will be saved for use in creating the video
    if i % step == 0:
        # Sets updated value into proper place in 3D array of U that holds 400 frames
        U_array[i//step] = U
        # Sets updated value into proper place in 3D array of V that holds 400 frames
        V_array[i//step] = V
        
        # This code properly plots and saves the images from the U array
        # Do not need to save the image for V because we only need to plot U for this project
        plt.imshow(U, interpolation='bicubic', cmap=plt.cm.jet)
        plt.savefig("imshow-{:03d}.png".format(i))
        plt.close()


# Sets title and save name for video
title = "Gray-Scott Model for U with F = 0.041 and k = 0.06"
saveName = "Gray-Scott Model (F=0.041, K=0.06)"
# Creates video from the 400 saved images/frames
output = CreateVideo(U_array, title, saveName, frames, interval, repeat, blit, show)

