#include <stdio.h>

void pp(unsigned char z){
    printf("%3d(%02x)\n",z,z);
}


unsigned char isZero(unsigned char x) {
    unsigned char ans;
    return ~(((x & (1<<7)) >> 7) |      // test bit 8
             ((x & (1<<6)) >> 6) |
             ((x & (1<<5)) >> 5) |
             ((x & (1<<4)) >> 4) |
             ((x & (1<<3)) >> 3) |
             ((x & (1<<2)) >> 2) |
             ((x & (1<<1)) >> 1) |
             (x & 1))                   // test bit 1
             & 1;              
}

unsigned char isEqual(unsigned char x, unsigned char y) {
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

unsigned char isLess(unsigned char x, unsigned char y) {
    return (!isEqual(x,y)) && (!isGreater(x,y));
}

unsigned char isNotEqual(unsigned char x, unsigned char y) {
    return !isEqual(x,y);
}

unsigned char isLessOrEqual(unsigned char x, unsigned char y) {
    return isEqual(x,y) || isLess(x,y);
}

unsigned char isGreaterOrEqual(unsigned char x, unsigned char y) {
    return isEqual(x,y) || isGreater(x,y);
}

unsigned char power2(unsigned char n){
    return (n==0) ? 0: (n &(n-1))==0;
}

unsigned short mult1(unsigned char n, unsigned char m) {
    unsigned char i;
    unsigned short ans = 0;
    if (n < m) {
        for(i=0; i < n; i++)
        ans += m;
    } else {
        for(i=0; i < m; i++)
        ans += n;
    }
    return ans;
}

unsigned short mult2(unsigned char n, unsigned char m) {
    unsigned char i;
    unsigned short ans = 0;
    for(i=0; i < 8; i++) {
        if (m & 1) {    
            ans += n << i;
        }
        m >>= 1;
    }
    return ans;
}

unsigned char div1(unsigned char n, unsigned char m, unsigned char *r) {
    unsigned char q=0;
    *r = n;
    while (*r > m) {
        q++;
        *r -= m;
    }
    return q;
}

unsigned char div2(unsigned char n, unsigned char m, unsigned char *r) {
    unsigned char i;
    *r =0;
    for (i=0; i < 8; i++) {
        *r = (*r << 1) + ((n & 0x80) !=0);
        n <<= 1;
        if ((*r-m)>=0){
            n|=1;
            *r -=m;
        }
    }
    return n;
}

unsigned char sqr(unsigned char n) {
    unsigned char c=0, p=1;
    while (n >= p) {
        n -= p;
        p += 2;
        c++;
    }
    return c;
}

// int main(){
//     // unsigned char z;

//     // z = 189 & 222;
//     // pp(z);
//     // z = 189 | 222;
//     // pp(z); 
//     // z = 189 ^ 222;
//     // pp(z);
//     // z = z ^ 222;
//     // pp(z);
//     // z =~z;
//     // pp(z);
    
//     // return 0;

//     // if (0xBD & 0x08) {
//     // printf("bit 3 on\n");
//     // }
//     // else {
//     //     printf("bit 3 off\n");
//     // }

//     // unsigned char x,y;
//     // x = 0xBD;
//     // y = 0xDE;
//     // isEqual(x,y) ? printf("equal\n") : printf("not equal\n");
//     // isEqual(x,x) ? printf("equal\n") : printf("not equal\n");
//     // isGreater(x,y) ? printf("x > y\n") : printf("x <= y\n");

//     unsigned char x, y, z;
//     x = 0xEE;
//     y = 0x35;
//     z = x + y;
//     printf("%x\n", z);

//     unsigned char n=123, m=7;
//     unsigned char q,r;
//     q = div1(n, m, &r);
//     printf("quotient=%d, remainder=%d\n", q,r);
//     return 0;
// }
int main(void) {
    unsigned char a = 0xBD, b = 0xDE, z;
    unsigned char q, r;

    // Bitwise ops
    z = a & b; printf("AND = "); pp(z);
    z = a | b; printf("OR  = "); pp(z);
    z = a ^ b; printf("XOR = "); pp(z);

    // Comparisons
    printf("isEqual(a,b) = %u\n", isEqual(a,b));
    printf("isGreater(a,b) = %u\n", isGreater(a,b));
    printf("isLess(a,b) = %u\n", isLess(a,b));

    // Power of two
    printf("power2(8) = %u\n", power2(8));
    printf("power2(10) = %u\n", power2(10));

    // Multiplication
    printf("mult1(20,14) = %hu\n", mult1(20,14));
    printf("mult2(20,14) = %hu\n", mult2(20,14));

    // Division
    q = div1(123, 7, &r);
    printf("div1(123,7): q=%u r=%u\n", q, r);
    q = div2(123, 7, &r);
    printf("div2(123,7): q=%u r=%u\n", q, r);

    // Addition / subtraction wrap
    printf("0xEE + 0x35 = 0x%02X\n", (unsigned char)(0xEE + 0x35));
    printf("0x00 - 0x01 = 0x%02X\n", (unsigned char)(0x00 - 1));

    // Square root
    printf("sqr(64) = %u\n", sqr(64));
    printf("sqr(50) = %u\n", sqr(50));

    return 0;
}
