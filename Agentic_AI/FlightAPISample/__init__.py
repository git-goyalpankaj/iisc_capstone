import UtilityClasses.dbUtils
import BLClasses.flight_booking_operation

cnx = UtilityClasses.dbUtils.connect_to_mysql()
connected = UtilityClasses.dbUtils.is_db_connected(cnx)
print("Successfully connected to mySql DB - ", connected)

def get_pnr_details():
    pnr_number = 'BIN1004'
    if BLClasses.flight_booking_operation.check_if_pnr_exists(pnr_number):
        result = BLClasses.flight_booking_operation.get_pnr_details(pnr_number)
        print(result)
        result = BLClasses.flight_booking_operation.cancel_ticket(pnr_number)
        print(result)
    else:
        print("Provided PNR doesn't exist. Please check.")

# Defining main function
def main():
    get_pnr_details()

# Using the special variable
# __name__
if __name__=="__main__":
    main()




