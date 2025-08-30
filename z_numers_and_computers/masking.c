#include <stdio.h>

void pp(unsigned char z){
    printf("%3d(%02x)\n",z,z);
}


unsigned char isZero(unsigned char x) {
unsigned char ans;
return ~(((x & (1<<7)) >> 7) | // test bit 7
((x & (1<<6)) >> 6) |
((x & (1<<5)) >> 5) |
((x & (1<<4)) >> 4) |
((x & (1<<3)) >> 3) |
((x & (1<<2)) >> 2) |
((x & (1<<1)) >> 1) |
(x & 1)) & 1;                  // test bit 1
}

unsigned char isEqual(unsigned char x,
unsigned char y) {
return isZero(x ^ y);
}

unsigned char isGreater(unsigned char a, unsigned char b) {
    unsigned  char x,y;
    x = ~a & b;
    y = a & ~b;

    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    
    return ~isZero(x & ~y) & 1;
}

unsigned char isLess(unsigned char x,
unsigned char y) {
return (!isEqual(x,y)) && (!isGreater(x,y));
}

unsigned char isNotEqual(unsigned char x,
unsigned char y) {
return !isEqual(x,y);
}

unsigned char isLessOrEqual(unsigned char x,
unsigned char y) {
return isEqual(x,y) || isLess(x,y);
}

unsigned char isGreaterOrEqual(unsigned char x,
unsigned char y) {
return isEqual(x,y) || isGreater(x,y);
}

int main(){
    // unsigned char z;

    // z = 189 & 222;
    // pp(z);
    // z = 189 | 222;
    // pp(z); 
    // z = 189 ^ 222;
    // pp(z);
    // z = z ^ 222;
    // pp(z);
    // z =~z;
    // pp(z);
    
    // return 0;

    // if (0xBD & 0x08) {
    // printf("bit 3 on\n");
    // }
    // else {
    //     printf("bit 3 off\n");
    // }

    // unsigned char x,y;
    // x = 0xBD;
    // y = 0xDE;
    // isEqual(x,y) ? printf("equal\n") : printf("not equal\n");
    // isEqual(x,x) ? printf("equal\n") : printf("not equal\n");
    // isGreater(x,y) ? printf("x > y\n") : printf("x <= y\n");

    unsigned char x, y, z;
    x = 0xEE;
    y = 0x35;
    z = x + y;
    printf("%x\n", z);
}