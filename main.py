import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./resources/sample.JPG')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_rgb.tolist()

plt.imshow(img_rgb)
plt.show()

# comparar ambas imágenes
def compare(original, manipulated, title_1="Original", title_2="Manipulada"):
    plt.figure(figsize=(15, 25))
    plt.subplot(1, 2, 1)
    plt.title(title_1)
    plt.imshow(original)
    plt.subplot(1, 2, 2)
    plt.title(title_2)
    plt.imshow(manipulated)
    plt.show()

# funciones que ayudan en algunos procesamientos más complejos
def print_shape(img):
    try:
        print("Las dimensiones del array son:", len(img), "x", len(img[0]), "x", len(img[0][0]))
    except:
        print("Las dimensiones del array son:", len(img), "x", len(img[0]))

def add_list(img1, img2):
    return [[img1[i][j] + img2[i][j] for j in range(len(img1[0]))] for i in range(len(img1))]

print("Original:")
print_shape(img_rgb)
print("\n")

def channel_first(img):
    return [[[img[j][k][i] for k in range(len(img[0]))] for j in range(len(img))] for i in range(len(img[0][0]))]

print("Después de channel_first")
z = channel_first(img_rgb)
print_shape(z)
print("\n")

def channel_last(img):
    return [[[img[k][i][j] for k in range(len(img))] for j in range(len(img[0][0]))] for i in range(len(img[0]))]

print("Después de channel_last")
z = channel_last(z)
print_shape(z)

# mostrar canales
channel_wise = channel_first(img_rgb)
plt.figure(figsize=(10, 20))
plt.subplot(1, 3, 1)
plt.title("Rojo")
plt.imshow(channel_wise[0], cmap='Reds')
plt.subplot(1, 3, 2)
plt.title("Verde")
plt.imshow(channel_wise[1], cmap='Greens')
plt.subplot(1, 3, 3)
plt.title("Azul")
plt.imshow(channel_wise[2], cmap='Blues')
plt.show()

# adición de canales
def channel_wise_addition(img):
    temp = channel_first(img)
    return add_list(temp[2], add_list(temp[0], temp[1]))

plt.imshow(channel_wise_addition(img_rgb), cmap="gray")
plt.show()

# invertir canales
def invert(img):
    return [[[255 - k for k in j] for j in i] for i in img]

compare(img_rgb, invert(img_rgb))

# voltear en vertical
def mirror_v(img):
    return [img[-i - 1] for i in range(len(img))]

compare(img_rgb, mirror_v(img_rgb))

#voltear en horizontal
def mirror_h(img):
    return [[img[i][-j - 1] for j in range(len(img[0]))] for i in range(len(img))]

compare(img_rgb, mirror_h(img_rgb))

# rotar izquierda
def rotate_left(img):
    return [[[img[j][-1 - i][k] for k in range(len(img[0][0]))] for j in range(len(img))] for i in range(len(img[0]))]

compare(img_rgb, rotate_left(img_rgb))

# rotar derecha
def rotate_right(img):
    return [[[img[-1 - j][i][k] for k in range(len(img[0][0]))] for j in range(len(img))] for i in range(len(img[0]))]

compare(img_rgb, rotate_right(img_rgb))

# añadir borde
def pad(img, width):
    padded = np.zeros([img.shape[0] + (width//2) + (width//2), img.shape[1] + (width//2) + (width//2)])
    padded[width//2 : padded.shape[0] - (width//2), width//2 : padded.shape[1] - (width//2)] = 255
    return padded

compare(img_rgb, pad(img_rgb, 25))

# desenfocar
def blur(img, strength=1):
    def blur_strength(img_aux):
        temp1 = []
        for i in range(len(img_aux)):
            temp2 = []
            for j in range(len(img_aux[0])):
                temp3 = []
                for k in range(len(img_aux[0][0])):
                    a_pixels = 1
                    temp = img_aux[i][j][k]
                    try:
                        temp += img_aux[i + 1][j + 1][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i + 1][j][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i + 1][j - 1][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i][j - 1][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i - 1][j - 1][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i - 1][j][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i - 1][j + 1][k]
                        a_pixels += 1
                    except:
                        True
                    try:
                        temp += img_aux[i][j + 1][k]
                        a_pixels += 1
                    except:
                        True

                    temp3.append(int(temp / a_pixels))
                temp2.append(temp3)
            temp1.append(temp2)
        return temp1

    temp = img.copy()
    for i in range(strength):
        temp = blur_strength(temp)
    return temp

compare(img_rgb, blur(img_rgb, 10))

# cambiar tamaño
def resize(img, size):
    return [[[img[int(len(img) * i / size[0])][int(len(img[0]) * j / size[1])][k] for k in range(3)]
             for j in range(size[1])] for i in range(size[0])]

compare(img_rgb, resize(img_rgb, (1000, 1000)))

# cambiar luminosidad
def lightness(img, b=50):
    return [[[int((255 * (b / 100)) + (img[i][j][k] * (1 - (b / 100)))) for k in range(len(img[0][0]))]
             for j in range(len(img[0]))] for i in range(len(img))]

compare(img_rgb, lightness(img_rgb, 25))

# cambiar brillo
def brightness(img, strength=0):
    return [[[int((510 / (1 + (2.7183 ** (-strength * img[i][j][k] / 255)))) - 255) for k in range(len(img[0][0]))]
             for j in range(len(img[0]))] for i in range(len(img))]

compare(img_rgb, brightness(img_rgb, 5))

# cambiar contraste
def contrast(img, strength=0):
    return [[[int(255 / (1 + (2.7183**(-strength * ((img[i][j][k] - 127.5) / 127.5))))) for k in range(len(img[0][0]))]
             for j in range(len(img[0]))] for i in range(len(img))]

compare (img_rgb, contrast(img_rgb, 5))

# aplicar aberración cromática
def chromatic_aberration_effect(img, strength=5, bgvalue=0):
    temp = channel_first(img)
    red = pad(channel_last([temp[0]]), (0, 2 * strength, 2 * strength, 0), bgvalue)

    green = pad(channel_last([temp[1]]), (strength, strength, strength, strength), bgvalue)

    blue = pad(channel_last(temp[2]), (2 * strength, 0, 0, 2 * strength), bgvalue)

    return channel_last(channel_first(red) + channel_first(green) + channel_first(blue))

compare(img_rgb, chromatic_aberration_effect(img_rgb, strength=10, bgvalue=0))
compare(img_rgb, chromatic_aberration_effect(img_rgb, strength=10, bgvalue=255))
