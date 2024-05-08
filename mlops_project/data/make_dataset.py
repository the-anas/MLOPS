import torch

if __name__ == '__main__':
    # Get the data and process it
    # load data, save it in  a single tensor, normalize it and save it to processed


    #load all images
    img1 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_0.pt")
    img2 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_1.pt")
    img3 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_2.pt")
    img4 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_3.pt")
    img5 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_4.pt")
    img6 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_images_5.pt")

    data = torch.cat((img1,img2, img3, img4, img5, img6), dim=0)

    # normalize
    mean = data.mean()
    std = data.std()

    # Normalize the tensor
    normalized = (data - mean) / std

    torch.save(normalized, r"C:\Users\anasn\OneDrive\Desktop\mlops\data\processed\processed_img.pt")


    # do the same thing for labels
    
    lab1 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_0.pt")
    lab2 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_1.pt")
    lab3 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_2.pt")
    lab4 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_3.pt")
    lab5 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_4.pt")
    lab6 = torch.load(r"C:\Users\anasn\OneDrive\Desktop\dtu_mlops\data\corruptmnist\train_target_5.pt")

    data_lab = torch.cat((lab1, lab2, lab3, lab4, lab5, lab6), dim=0)
    torch.save(data_lab, r"C:\Users\anasn\OneDrive\Desktop\mlops\data\processed\processed_lab.pt")  