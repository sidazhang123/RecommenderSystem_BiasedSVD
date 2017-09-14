import math
import random
import pickle
import matplotlib.pyplot as plt
import mysql.connector
import subprocess


def Average(fileName):         # Mean of all existing scores in training data
    fi = open(fileName, 'r')
    result = 0.0
    count = 0
    for line in fi:
        count += 1
        arr = line.split()
        result += int(arr[2].strip())
    return result / count


def InnerProduct(v1, v2):      # of two vectors
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result


def PredictScore(av, bu, bi, pu, qi):     # according to trained model
    pScore = av + bu + bi + InnerProduct(pu, qi)
    if pScore < 1:
        pScore = 1
    elif pScore > 5:
        pScore = 5


    return pScore


def SVD(configureFile, trainDataFile, modelSaveFile,testDataFile):
    # Read configuration file
    print("Initializing...")
    fi = open(configureFile, 'r')
    line = fi.readline()
    arr = line.split()
    averageScore = Average(testDataFile)
    learnRateDecay=float(arr[0].strip())
    userNum = int(arr[1].strip())
    itemNum = int(arr[2].strip())
    factorNum = int(arr[3].strip())  # Customized num of latent factors

    learnRate = float(arr[4].strip())
    penalty = float(arr[5].strip())  # Regularization
    maxIter = int(arr[6].strip())
    fi.close()


    bi = [0.0 for i in range(itemNum)]  # Init bias items for 'items' and 'users'
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]  # Init factorized matrix filled
    pu = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(userNum)]
    print("Initialization completed.")

    debug=True if 'y'==input('Show Error of every loop? (y/n)\n') else False

    print("Training...")

    # train model
    preRmse = 100.0
    plot_step=[]
    plot_RMSE=[]
    plot_MAE=[]

    for step in range(maxIter):
        plot_step.append(step)
        fi = open(trainDataFile, 'r')
        for line in fi:
            arr = line.split()
            uid = int(arr[0].strip())
            iid = int(arr[1].strip())
            score = int(arr[2].strip())
            prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])

            eui = score - prediction # Err of prediction

            # bu ← bu + γ ∙ (erru,i − α ∙ bu)
            bu[uid] += learnRate * (eui - penalty * bu[uid])
            # bi ← bi + γ ∙ (erru,i − α ∙ bi)
            bi[iid] += learnRate * (eui - penalty * bi[iid])
            for k in range(factorNum):
                temp = pu[uid][k]
                # pu ← pu + γ ∙ (erru,i ∙ qi − α ∙ pu)
                pu[uid][k] += learnRate * (eui * qi[iid][k] - penalty * pu[uid][k])
                # qi ← qi + γ ∙ (erru,i ∙ pu − α ∙ qi)
                qi[iid][k] += learnRate * (eui * temp - penalty * qi[iid][k])
        fi.close()
        learnRate *= learnRateDecay # Gradually reduce learning rate for accurate converging. But this can slow the process.
        curRmse, mae = Validate(testDataFile, averageScore, bu, bi, pu, qi) # Compute RMSE to testing data in every iteration
        if debug:
            print("test_RMSE in step %d: %f" %(step, curRmse))             # to evaluate iteration number
            print("test_MAE: %f\n" % mae)

        if curRmse >= preRmse:
            plot_step=plot_step[:-1]
            break
        else:
          preRmse = curRmse
        plot_MAE.append(mae)
        plot_RMSE.append(preRmse)

    # write the model to files
    with open(modelSaveFile, 'wb')as fo:
        pickle.dump(bu, fo)
        pickle.dump(bi, fo)
        pickle.dump(qi, fo)
        pickle.dump(pu, fo)
        pickle.dump(averageScore,fo)
    #plot
    if not debug:

        print("Total RMSE : %f" %curRmse)
        print("Total MAE : %f" % mae)
        print('=======================================')
    plt.figure(figsize=(16,9))
    plt.title('Num of iterations Vs test RMSE & MAE')
    RMSE, = plt.plot(plot_step,plot_RMSE, label='RMSE')
    MAE, = plt.plot(plot_step, plot_MAE, label='MAE')
    plt.legend(handles=[RMSE, MAE])
    plt.ylim((min(min(plot_RMSE), min(plot_MAE)) - 0.05, max(max(plot_RMSE), max(plot_MAE))+0.05))
    # plt.show()
    print("Training completed.")
    if 'y'==input('Show graph of Errors? (y/n)\n'):
        plt.show()



def Validate(testDataFile, av, bu, bi, pu, qi): # Get each step's performance while training
    count = 0
    rmse = 0.0
    mae=0.0
    fi = open(testDataFile, 'r')
    for line in fi:
        count += 1
        arr = line.split()
        uid = int(arr[0].strip())
        iid = int(arr[1].strip())
        pScore = PredictScore(av, bu[uid], bi[iid], pu[uid], qi[iid])

        tScore = int(arr[2].strip())
        rmse += (tScore - pScore) * (tScore - pScore)
        mae+=abs(tScore - pScore)
    fi.close()

    return math.sqrt(rmse / count),mae/count


# use trained model to predict
def Predict(modelSaveFile, testDataFile,d):
    print("Predicting ...")
    cnx = mysql.connector.connect(user='admin', database='recsys', password='admin')
    cursor = cnx.cursor()
    cursor.execute("TRUNCATE scores_u"+str(d))
    exe = "INSERT INTO scores_u" + str(d) + "(user_id,item_id, rating) VALUES (%s, %s,%s)"
    add_rating = (exe)


    # get model
    with open(modelSaveFile, 'rb') as fi:
        bu = pickle.load(fi)
        bi = pickle.load(fi)
        qi = pickle.load(fi)
        pu = pickle.load(fi)
        averageScore = pickle.load(fi)


    # Predict and store results to db
    fi = open(testDataFile, 'r')

    for line in fi:
        arr = line.split()
        uid = int(arr[0].strip())
        iid = int(arr[1].strip())
        pScore = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
        rating_data=(uid,iid,pScore)
        cursor.execute(add_rating, rating_data)

    fi.close()

    cnx.commit()
    cursor.close()
    cnx.close()
    print("Predicting completed.\n Results saved to DB.")

def getRecommendation(k='',uid='',res='',q=''):
    cnx = mysql.connector.connect(user='admin', database='recsys', password='admin')
    cursor = cnx.cursor()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if q!='':
        try:
            cursor.execute(q)
            for i in cursor:
                print(i)
            return
        except:
            print("Error happened in SQL querying.")
            return
    if uid!='':
        query = "SELECT rating, title FROM (scores_u{:} natural join titles) WHERE user_id = {:} ORDER BY rating DESC ".format(res,uid)
        if k!='':
            query+=" LIMIT "+str(k)

        cursor.execute(query)
        print(" ===  Recommend to User "+uid+" ===")
        print("RATING\t\t\tTITLE")

        for (rating,title) in cursor:
            print("{:}\t{:}".format(rating,title))
    else:
        query = "SELECT user_id, rating, title FROM (scores_u"+str(res)+" natural join titles) ORDER BY rating DESC"
        if k!='':
            query+=" LIMIT "+str(k)
        cursor.execute(query)
        print("USER\tRATING\t\t\tTITLE")
        for (user_id,rating,title) in cursor:
            print("{:}\t\t{:}\t\t{:}".format(user_id,rating,title))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
def ui():
    a=True
    b=True
    c=True
    ddd=True

    while True:
        while a:
            print("Configuring ... ")
            if 'y'==input('Skip Conf ?(y\Enter)\n'):
                break
            configureFile='svd.conf'
            trainDataFile = input("Path to training data ?\n")
            testDataFile = input("Path to testing data ?\n")
            d=input("Path to save MODEL ? 'Enter' to ignore.\n")
            modelSaveFile = d if d!='' else 'svd_model.dat'
            if 'y'==input("Check or Modify parameters for training? (y/n)\n"):
                subprocess.call('nano svd.conf', shell=True)
            a=False
        while b:
            print('Prepare for training.')
            if 'y'==input('Skip Training ?(y\Enter)\n'):
                break
            SVD(configureFile, trainDataFile, modelSaveFile, testDataFile)
            b=False
        while c:
            if 'y'==input('Skip Predicting ?(y\Enter)\n'):
                break

            for dd in testDataFile[::-1]:
                if dd.isdigit():
                    Predict(modelSaveFile,testDataFile,dd)
                    break

            c=False
        while ddd:
            if 'y'==input('Skip Recommending ?(y\Enter)\n'):
                break
            print(" == Recommendation List ==")
            uid=input("For which User? (id) 'Enter' to show All.\n")
            k= input('How many rows to view? From the top rating. "Enter" to show All.\n')
            q=input('sql? "Enter" to skip.')
            for dd in testDataFile[::-1]:
                if dd.isdigit():
                    getRecommendation(k, uid, dd, q)
                    break

            ddd=False

        a = True
        b = True
        c = True
        ddd = True

if __name__ == '__main__':
    ui()

