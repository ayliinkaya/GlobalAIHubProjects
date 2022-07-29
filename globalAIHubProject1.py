"""Proje 1
Bu projede bir öğrenci not sistemi oluşturacaksınız. Sizden istenilenler:"""

# ● Kendinize bir ders belirleyiniz. (Matematik,Fizik, Lineer Cebir vb.)
# ● Not aralığınızı oluşturunuz (100-80 ⇒ A, 79-70 ⇒ B vb.)
# ● Öğrenci Bilgilerini (Ad, Soyad, Okul No, sınav puanı) girebileceğiniz ve bu
# bilgilerin tutulabileceği bir sistem oluşturunuz.
# ● Girilen bilgilerden yola çıkarak öğrencinin dersi geçip geçmediğini göstermesi
# gerekmektedir.
# ● Öğrenci dersi geçti ise öğrencinin bilgilerinin tutulduğu alana “Geçti” yazısı,
# öğrenci dersi geçemedi ise “kaldı” yazısını göstermesi gerekmektedir.
# ● Notları girilen öğrencilerden dersi geçenleri ve geçmeyenleri gösteren bir
# Dataframe oluşturunuz.
# ● Oluşturulan Dataframe’i Excel tablosuna dönüştürünüz.

student_scores = [84,76,92]
total_score = (student_scores[0] * 0.2) + (student_scores[1] * 0.2) + (student_scores[2] * 0.6)

if (90 <= total_score <= 100):
    print("Pass!\nPerformance Designation: Excellent!\nLetter grade: A")

elif (80 <= total_score <= 89):
    print("Pass!\nPerformance Designation: Good!\nLetter grade: B")

elif (70 <= total_score <= 79):
    print("Pass!\nPerformance Designation: Fair!\nLetter grade: C")

elif (60 <= total_score <= 69):
    print("Pass!\nPerformance Designation: Weak!\nLetter grade: D")

elif total_score < 60:
    print("Fail!\nPerformance Designation: Fail!\nLetter grade: F")


def registration(student_name, student_no, total_score):
    student_list = []
    student_list.append(student_name)
    student_list.append(student_no)
    student_list.append(total_score)
    result = "Pass"
    if total_score < 60:
        result = "Fail"
    student_list.append(result)
    return student_list


def student_registration():
    student_name = input("Please enter student's name: ")
    student_no = input("Please enter student no: ")
    total_score = int(input("Student's total score: "))
    return student_name, student_no, total_score

student_reg_list = [["Alican Kayacan", "2612548963", 89, "Pass"],
                    ["İdil Sağlam", "1245965785", 95, "Pass"],
                    ["İsmail Gözcü", "7851249634", 11, "Fail"]]

import pandas as pd

x, y, z = student_registration()
student_reg_list.append(registration(x, y, z))
dataframe = pd.DataFrame(student_reg_list, columns = ["student_name", "student_no", "total_score", "result"])
dataframe

dataframe.to_excel("new_student.xlsx",sheet_name="1",index=False)
pd.read_excel("new_student.xlsx")

