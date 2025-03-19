class Car:
    color = "red" # Class attribute
    def __init__(self, number_wheels, type):
        self.wheels = number_wheels #O bject attributes
        self.type = type
    def get_wheels_num(self):
        return self.wheels
    def get_type(self):
        return self.type

if __name__ == "__main__":
    honda1 = Car(4, "accord")
    honda2 = Car(2, "crf250")
    print(f"First honda num of wheels: {honda1.get_wheels_num()} \nFirst honda type: {honda1.get_type()}\n\n")
    print(f"Second honda num of wheels: {honda2.get_wheels_num()} \nSecond honda type: {honda2.get_type()}\n\n")
    print(f"Are there colors the same? {honda1.color is honda2.color}\n")
    print(f"Honda1 Color: {honda1.color}\n")
    print(f"Honda2 Color: {honda2.color}")

    # Change the color
    Car.color = "Blue"
    print(f"Are there colors the same? {honda1.color is honda2.color}\n")
    print(f"Honda1 Color: {honda1.color}\n")
    print(f"Honda2 Color: {honda2.color}")