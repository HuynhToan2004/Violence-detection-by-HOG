from data_preprocessing import load_data, process_video
from train import train_and_evaluate

def main():
    X, y = load_data('/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset')
    
    a = [] 
    b = []  

    for i, j in zip(X, y):
        features, labels = process_video(i, j)
        a.append(features)
        b.append(labels)

    c = [label[0] for label in b] 
    
    train_and_evaluate(a, c)

if __name__ == "__main__":
    main()
