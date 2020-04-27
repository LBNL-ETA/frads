from frads import makesky
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modified version of epw2wea with added\
     capability of including only the daylight hours")
    parser.add_argument('-a', help='latitude')
    parser.add_argument('-o', help='longitude')
    parser.add_argument('-z', help='zipcode (U.S. only)')
    parser.add_argument('-dh', action="store_true", help='output only for daylight hours')
    parser.add_argument('-sh', type=float, help='start hour (float)')
    parser.add_argument('-eh', type=float, help='end hour (float)')
    args = parser.parse_args()

    if args.dh: print("Writing only daylight hours ...")

    if args.z is not None:
        epw = makesky.getEPW.from_zip(args.z)
    else:
        epw = makesky.getEPW(args.a, args.o)
    wea = makesky.epw2wea(epw=epw.fname, dh=args.dh, sh=args.sh, eh=args.eh)
    print(wea.wea)
