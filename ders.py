"""import pandas as pd
veri=pd.DataFrame()
print(veri)"""
"""import pandas as pd"""
"""liste=[[221,"Ahmet Taşpınar","İstanbul"],
       [929,"Selkan Naimoğlu","Kayseri"],
       [222,"Ayşe Öztürk","Antalya"]]
df=pd.DataFrame(liste,columns=["Öğrenci No","Adı Soyadı","İl"])
print(df)"""
"""df=pd.DataFrame({
    "Öğrenci No":[221,347,222],
    "Adı Soyadı":["Ahmet Taşpınar","Selkan Naimoğlu","Ayşe Öztürk"],
    "İl":["İstanbul","Kayseri","Antalya"]})
print(df)"""
"""hasta=[["Eser Kaya",10,15],
       ["Yasin Elverdi",8.5,13],
       ["Kerem Ala",7.2,11]]
df=pd.DataFrame(hasta,
  columns=["Hasta Adı","Küçük Tansiyon","Büyük Tansiyon"])
print(df)"""
"""liste=[["Burak",32,"İstanbul"],
       ["Selkan",40,"Kayseri"],
       ["Ata",4,"Isparta"],
       ["Yaman",1,"Isparta"]]
df=pd.DataFrame(liste,columns=["Adı","Yaş","İl"])"""
"""print(df)"""
"""print(df["Adı"])"""
"""print(df.iloc[:,2])"""
"""print(df.iloc[2])"""
"""print(df.iloc[2:4])"""
"""print(df.iloc[2:4,2])"""
"""df.iloc[3,1]=2
print(df)"""
"""df["Okul"]=pd.Series(["Üniversite","Üniversite","Ana Okulu","Okulu yok"])"""
"""del df["İl"]"""
"""df=df.drop(3)
print(df)"""
"""uzaklıklar=[
["İstanbul-Adana",939],
["İstanbul-Afyon",460],
["İstanbul-Ankara",453],
["İstanbul-Ankara",724],
["İstanbul-Balıkesir",390]]
df=pd.DataFrame(uzaklıklar,columns=["İller","Uzaklıklar(Km)"])"""
"""print(df)"""
"""yeni_uzaklık=["İstanbul-Çanakkale",320]
yeni_uzaklık=pd.DataFrame([yeni_uzaklık],
    columns=["İller","Uzaklıklar(Km)"])"""
"""df=pd.concat([yeni_uzaklık,df],ignore_index=True)"""
"""df=pd.concat([df,yeni_uzaklık],ignore_index=True)
print(df)"""
"""öğrenci=[["Selim",25,60,50],
 ["Yasin",60,80,70],
 ["Sedat",60,65,55],
 ["Deniz",70,85,80],
 ["Derya",95,90,80]]
df=pd.DataFrame(öğrenci,columns=["Adı","Matematik","Fizik","Kimya"])"""
"""df=df.sort_values(by='Adı')"""
"""df=df.sort_values(by='Kimya')"""
"""df=df.sort_values(by='Adı',ascending=False)
print(df)"""

"""x=23
if x>0:
    print("Sayı Pozitiftir")"""
"""x=45
if x<5.5:
    print("Sayı pozitiftir")"""
"""x=0
if x>0:
    print("Sayı pozitiftir")
elif x<0:
    print("Sayı negatiftir")
else:
    print("Sayı sıfır değerlidir")"""
"""a=715;b=713;c=711
if a<b:
    if b<c:
        küçük=a
    elif a<c:
        küçük=a
    else:
        küçük=c
elif b<c:
    küçük=b
else:
    küçük=c
print("En küçük sayı:",küçük)"""
"""import math
a=1;b=-3;c=2
diskriminant=b**2-4*a*c
if diskriminant>0:
    x1=(-b+math.sqrt(diskriminant))/2*a
    x2=(-b-math.sqrt(diskriminant))/2*a
    print("birinci kök:",x1)
    print("ikinci kök",x2)
elif diskriminant<0:
        print("gerçek kök yok")
else:
            x1=-b/2*a
            print("eşit iki kök:",x1)"""
"""for x in range(1,5):
    print(x)"""
"""sayılar=[5,3,1,8,9,16,22,1,5]
for x in sayılar:
    if x<10:
        print(x)"""
"""dizi=[5,3,1,8,9,16,22,1,5]
for x in dizi:
    if (x>20):break
    print(x)"""
"""dizi=[25,29,36,44,30]
for x in dizi:
    if (x==29):continue
    print(x)"""
"""x=0
while x<6:
    print(x)
    x+=1"""
"""liste=[12,15,11,7,9,13,25,14,3,14]
toplam=0
eleman_sayısı=0
for eleman in liste:
    toplam=toplam + eleman
    eleman_sayısı=eleman_sayısı+1
ortalama=toplam/eleman_sayısı
print("Ortalama :",ortalama)"""
"""liste=[12,15,11,7,9,13,25,14,3,14]
for eleman in liste:
    if eleman < 10:"""
"""for eleman in liste:
    if eleman < 15:
        print(eleman)"""
"""for x in range(1,10):
    if x % 2 !=0:
        print(x, "sayısı tektir")
    else:
        print(x, "sayısı çifttir")"""
"""for x in range(0,10):
    if x==3 or x==6:
      continue
    print(x)"""
"""import math as m
for x in range(50,60):
    karekök=m.sqrt(x)
    print(x,"sayısının karekökü=",karekök)"""
    































"""def test():
    print("Merhaba Türkiye")
test()"""
"""def hesaplama():
    sayı1=25
    sayı2=20
    ortalama=(sayı1 + sayı2)/2
    print(ortalama)
hesaplama()"""
"""def mesaj():
    mesaj_görüntüle()
def mesaj_görüntüle():
    print("Merhaba Selkan")
mesaj()"""    
"""def mesaj():
    mesaj_görüntüle()
    print("Merhaba Türkiye")
def mesaj_görüntüle():
    print("Merhaba Selkan")
mesaj()"""
"""def işlem():
    sayı1=25
    sayı2=65
    ortalama=(sayı1+sayı2)/2
    return(ortalama)
y=işlem()
print(y)"""
"""def işlem():
    sayı1=25
    sayı2=65
    return(sayı1,sayı2)
y=işlem()
ortalama=(y[0] + y[1])/2
print(ortalama)"""
"""def işlem():
    liste=[15,9,11,20,25,30,34,25]
    return(liste)
y=işlem()
print(y)"""
"""import pandas as pd
def df_işlem():
    d={'İsimler':["Begüm","Ata","Yaman"],
       'Matematik puan':[78,96,89],
       'Fizik puan':[83,75,77]}
    df=pd.DataFrame(data=d)
    return(df)
y=df_işlem()
print(y)"""
"""def test_işlemi(x):
    if (x>0) : mesaj="Pozitif sayı"
    else:
        if(x<0):mesaj="Negatif sayı"
        else:mesaj="Sıfır"
    return(mesaj)
y=test_işlemi(-30)
print(y)"""
"""def ortalama_işlemi(sayılar):
    toplam=sum(sayılar)
    ortalama=toplam/len(sayılar)
    return(ortalama)
liste=[25,30,45,50,49]
y=ortalama_işlemi(liste)
print(y)"""
"""import numpy as np
def sinüs(derece):
    radyanı=np.radians(derece)
    sonuç=np.sin(radyanı)
    return(sonuç)
y=sinüs(30)
print(y)"""
"""def tümsayılar(baştaki,sondaki):
    for x in range(baştaki,sondaki):
        print(x)
tümsayılar(3,10)"""
"""def tekçift(sayı):
    if sayı%2!=0:
        print(sayı,"sayısı tektir")
    else:
        print(sayı,"sayısı çifttir")
tekçift(10)"""
"""def karşılaştır(liste,sayı):
    for eleman in liste:
        if eleman<sayı:
            print(eleman)
x=[12,15,11,7,9,13,25,14,3,14]
karşılaştır(x,10)"""
"""def ortalama(liste):
    toplam=0
    eleman_sayısı=0
    for eleman in liste:
        toplam=toplam+eleman
        eleman_sayısı=eleman_sayısı+1
        ortalama=toplam/eleman_sayısı
    print("Ortalama:",ortalama)
liste=[12,15,11,7,9,13,25,14,3,14]
ortalama(liste)"""
"""import math as m
def karekökhesapla(sayı1,sayı2):
    for x in range(sayı1,sayı2):
        karekök=m.sqrt(x)
        print(x,"sayının karekökü=",karekök)
karekökhesapla(0,10)"""        
"""import numpy as np
liste=[45,56,21,25,70,52]
vektör=np.array(liste)
print(type(vektör))"""
"""import numpy as np
vektör=np.arange(0,11)
print(vektör)"""
"""import numpy as np
vektör=np.arange(0,11,2)
print(vektör)"""
"""import numpy as np
vektör=np.arange(0.25,5)
print(vektör)"""
"""import numpy as np
vektör=np.arange(0.25,5,1.5)
print(vektör)"""
"""import numpy as np
vektör=np.array([125,350,520,175])
print(vektör)"""
"""import numpy as np
vektör=np.array([[125],[350],[520],[175]])
print(vektör)"""
"""import numpy as np
vektör=np.array([3,6,3,2,1,7,9])"""
"""boyut=vektör.shape
print(boyut)"""
"""import numpy as np
vektör=np.array([6,3,7,8,1,0,3])
boyut=vektör.shape
print(boyut)"""
"""import numpy as np
vektör=np.array([6,3,7,8,1,0,3,5])
vektör=vektör.reshape((4,2))
print(vektör)"""
"""import numpy as np
vektör=np.array([3,6,3,2,1,7,9])"""
"""print(vektör[0])"""
"""print(vektör[3])"""
"""print(vektör[5])"""
"""uzunluk=len(vektör)
print(uzunluk)"""
"""import numpy as np
vektör=np.array([3,6,3,2,1,7,9])"""
"""vektör[6]=16"""
"""vektör[6]=20
print(vektör)"""
"""vektör[2]=55
print(vektör)"""
"""import numpy as np
vektör1=np.array([6,3,7,8,1,0,3])
vektör2=np.array([5,11,13,15,6,8,3])"""
"""print(vektör1+vektör2)"""
"""print(vektör2-vektör1)"""
"""print(vektör1*vektör2)"""
"""print(vektör1/(1+2*vektör2))"""
"""birleştirilmiş=np.concatenate((vektör1,vektör2))
print(birleştirilmiş)"""
"""import numpy as np
vektör=np.array([6,3,7,8,1,0,3])
toplam=sum(vektör)
print(toplam)"""
"""import numpy as np
vektör=np.array([6,3,7,8,1,0,3])"""
"""küm_toplam=np.cumsum(vektör)
print(küm_toplam)"""
"""sıralanmış=np.sort(vektör)
print(sıralanmış)"""
"""import numpy as np
vektör=np.array([7,4,5,5,8,3,5])
seçilen=np.where(vektör==5)
print(seçilen)"""
"""import numpy as np"""
"""vektör=np.array(["Burak","Begüm","Ata","Yaman"])
seçilen=np.where(vektör != "Burak")
print(seçilen)"""
"""vektör=np.array([6,3,7,8,1,0,3,5])
sorgu=vektör[vektör>5]
print(sorgu)"""
"""matris=[[25,16,-20,15],[17,25,33,21],[9,-55,48,36]]
print(matris)"""
"""matris = [
    [25,16,-20,15],
    [17,25,33,21],
    [9,-55,48,36]]
matris[1][2]=99
print(matris)"""
"""import numpy as np"""
"""mat=np.array([[5,3,1],[7,2,2],[1,0,6]])
print(mat)"""
"""mat=np.matrix("5,3,1;7,3,3;8,3,1")
print(mat)"""
"""import numpy as np
mat=np.matrix([[5,3,1],[7,2,2],[1,0,6]])
print(mat)"""
"""import numpy as np
sıfırla=np.zeros([3,3])
print(sıfırla)"""
"""doldur=np.full((3,3),1)
print(doldur)"""
"""import numpy as np
sıfırla=np.full((3,3),0)
print(sıfırla)"""
"""import numpy as np
mat1=np.matrix([[15,20,-5],[3,12,14],[0,18,15]])"""
"""print(mat1)"""
"""mat2=np.matrix([[7,3,2],[5,8,1],[-3,6,6]])"""
"""print(mat2)"""
"""mat3=mat1+mat2
print(mat3)"""
"""mat1=np.matrix([[3,7],[5,1],[-1,6]])"""
"""print(mat1)"""
"""mat2=np.matrix([[6,6],[9,2],[4,0]])"""
"""print(mat2)"""
"""mat3=mat1-mat2
print(mat3)"""
"""mat=np.matrix([[15,20,-5],[3,12,14],[0,18,15]])
print(mat)"""
"""toplam=np.sum(mat)
print(toplam)"""
"""import pandas as pd"""
"""veri=pd.read_table("C:/test/isimler.txt",delim_whitespace=True,
names=["Öğrenciler","Ders"])
print(veri)"""
"""df=pd.DataFrame({"Öğrenci No":[221,345,222],
  "Adı Soyadı":["Ahmet Taşpınar","Ata Basmacı","Ayşe Öztürk"],
  "İl":["İstanbul","Isparta","Antalya"]})"""
"""df.to_csv("C:/test/isim_soyad.csv",encoding='utf-8-sig')"""
"""import openpyxl
df.to_excel("C:/test/isim_soyad.xlsx", index=False)"""
"""veri=pd.read_excel("C:/test/isim_soyad.xlsx",index_col=0)
print(veri)"""
"""veri = pd.read_csv(" https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  names=["sepal_length",
  "sepal_width"
  "petal_length",
  "petal width",
  "class"]
  )
print(veri)"""
"""veri = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
 names = ["sepal_length",
 "sepal_width",
 "petal_length",
 "petal width",
 "class"]
 )                
print(veri)"""
"""import matplotlib.pyplot as plt
import pandas as pd"""
"""liste=[["Halit",8,9,7],
["Didem",5,9,9],
["Dilek",10,8,7],
["Aslan",9,10,9],
["Mert",8,9,10]]

df=pd.DataFrame(liste,
columns=["Öğrenci","Matematik","Fizik","Kimya"])"""

"""print(df)"""
"""df.plot(kind='scatter',x='Matematik',y="Fizik")
plt.title('Matematik-Fizik dersi puan karşılaştırmaları')
plt.show()"""
"""df.plot(kind='bar',x='Öğrenci',y='Matematik')
plt.title('Matematik dersi notları')
plt.ylabel('Başarı Puanı')
plt.show()"""
"""df.plot(kind='line',x='Öğrenci',y='Matematik')
plt.title('Matematik dersi notları')
plt.ylabel('Başarı puanı')
plt.show()"""
"""ortak=plt.gca()

df.plot(kind='line',x='Öğrenci',y='Matematik',ax=ortak)
df.plot(kind='line',x='Öğrenci',y='Fizik',ax=ortak)
df.plot(kind='line',x='Öğrenci',y='Kimya',ax=ortak)

plt.title('Öğrenci Başarı Puanları')
plt.ylabel('Başarı Puanı')
plt.show()"""
"""veri=pd.DataFrame({
   'Medeni Durum':['Hiç evlenmedi','Evli','Boşandı','Eşi öldü'], 
   'Sayı': [15.99,37.22,1.98,3.18]
})"""
"""print(veri)"""
"""#Grafik çizdir
veri.plot.pie(
    y='Sayı',
    labels=veri['Medeni Durum'],
    figsize=(5,5),
    autopct="%.1f%%"
)
plt.title("Türkiye'de medeni duruma göre nüfus dağılımı")
plt.ylabel("") #Y eksenini gizle
plt.show()"""
"""liste=[
["2015",5400],
["2016",6000],
["2017",6500],
["2018",6350],
["2019",7000]]

df=pd.DataFrame(liste,columns=["Yıllar","Satışlar"])
print(df)

df.plot(kind='line',x='Yıllar',y='Satışlar')
plt.title('Yıllara göre satışlar')
plt.ylabel('Satışlar')
plt.show()"""
"""import numpy as np"""
"""#Saat bazında hasta giriş sayısı (örnek veri)
saatler=np.arange(7,23) #07:00 - 23:00 saatleri
hasta_sayisi=np.random.randint(5, 50, len(saatler))

#Grafik çizimi
plt.figure(figsize=(10, 5))
plt.plot(saatler, hasta_sayisi,marker="o", linestyle="-", color="blue",
     label="Hasta Sayısı")
plt.xlabel("Saat")
plt.ylabel("Hasta Sayısı")
plt.title("Günlük Hasta Akışı")
plt.grid(True)
plt.legend()
plt.xticks(saatler, rotation=45)
plt.show()"""
"""# Hastane bölümleri ve hasta sayıları
bolumler=["Acil", "Ortopedi", "Kardiyoloji", "Nöroloji", "Dahiliye",
"Göz", "KBB"]
hasta_sayisi=np.random.randint(20, 150, len(bolumler))

#Grafik çizimi
plt.figure(figsize=(10, 5))
plt.bar(bolumler, hasta_sayisi, color="red", alpha=0.7)
plt.xlabel("Hastane Bölümleri")
plt.ylabel("Hasta Sayısı")
plt.title("Hatane Bölümlerine Göre Hasta Yoğunluğu")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()"""
"""#Bölümlere göre yatak doluluk oranları (%)
bolumler=["Acil","Kardiyoloji","Ortopedi","Dahiliye","Yoğun Bakım"]
doluluk_oranlari=[80,60,70,50,90] #% değerleri

#Grafik çizimi
plt.figure(figsize=(8,8))
plt.pie(doluluk_oranlari,labels=bolumler,autopct="%1.1f%%",
startangle=90,colors=["red","blue","green","purple","orange"])
plt.title("Hastane Yatak Doluluk Oranları")
plt.show()"""
"""#Randevu bekleme süreleri(dakika)-Ortalama 30 dk, standart sapma 10 dk
bekleme_sureleri=np.random.normal(30,10,200)

#Grafik çizimi
plt.figure(figsize=(10,5))
plt.hist(bekleme_sureleri,bins=20,color="green",alpha=0.7,edgecolor="black")
plt.axvline(np.mean(bekleme_sureleri),color="red",linestyle="dashed",
   linewidth=2,label=f"Ortalama:{np.mean(bekleme_sureleri):.1f}dk")
plt.xlabel("Bekleme Süresi (Dakika)")
plt.ylabel("Hasta Sayısı")
plt.title("Randevu Bekleme Süresi Dağılımı")
plt.legend()
plt.grid(axis="y",linestyle="--",alpha=0.7)
plt.show()"""






























































































































"""def enbüyük(a,b,c):
    if a>b:
        if a>c:
            mesaj="a en büyük sayıdır"
        else:
            mesaj="c en büyük sayıdır"
    elif b>c:
        mesaj="b en büyük sayıdır"
    else:
        mesaj="c en büyük sayıdır"
    return(mesaj)
#Ana kodlar
sonuç=enbüyük(2,1,3)
print(sonuç)"""
"""def faktöriyel(sayı):
    faktöriyel=1
    for i in range(1,sayı+1):
        faktöriyel=faktöriyel*i
    return(faktöriyel)

sonuç=faktöriyel(5)
print(sonuç)"""
"""import math as m
def ikinci(a,b,c):
    D=b^2-4*a*c
    if D>0:
        x1=(-b+m.sqrt(D))/2*a
        x2=(-b-m.sqrt(D))/2*a
        return(x1,x2)
    elif D==0:
        x1=b/2*a
        x2=x1
        return(x1,x2)
    else:
        print("Gerçek kök yok")

sonuç=ikinci(1,2,-1)
print("Kökler:",sonuç)"""
    







"""def deneme(veri,a):
    for index in range(0,len(veri),a):
        print(veri[index])
dizi=[19,25,36,37,44,48,56,65,73]
deneme(dizi,3)"""
"""def f(veri):
    index=1
    while index < len(veri):
        print(veri[index])
        index*=1
#fonksiyon çağırma
liste=[19,25,36,37,44,48,56,65,73]
f(liste)"""

"""def çarpma(liste):
    sonuç=liste[0]*liste[0]
    print(sonuç)
çarpma([4,5,6,8])"""    
"""def doğrusal_algo(liste):
    for item in liste:
        print(item)
doğrusal_algo([4,5,6,8,10])"""
"""import matplotlib.pyplot as plt
import numpy as np
x=[2,4,6,8,10,12]
y=[2,4,6,8,10,12]

plt.plot(x,y,'b')
plt.xlabel('Girişler')
plt.ylabel('Adımlar')
plt.title('Doğrusal karmaşıklık')
plt.show()"""
"""def doğrusal(liste):
    for x in liste:
        print(x)
        
    for x in liste:
        print(x)

doğrusal([4,5,6,8])"""    
"""def quatratic(liste):
    for x in liste:
        for xx in liste:
            print(x,' ',x)
quatratic([4,5,6,8])"""
"""def sayilari_yazdir(n):
    for i in range(n):  #n kez çalışır
        print(i,end=" ") #Sonucu yatay yazar
#Çağırma örneği:
sayilari_yazdir(4)
sayilari_yazdir(5)"""
"""def tum_eşleri_yazdir(n):
    for i in range(n):
        for j in range(n):  #Dış döngü: n kez
            print(f"{i}-{j}",end=" ")"""
"""yatay_dizi = [1,3,5,7,8]
for eleman in yatay_dizi:

 print(eleman)"""



            













        

    
