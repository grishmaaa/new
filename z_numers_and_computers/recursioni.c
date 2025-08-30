#include <stdio.h>
// #include <string.h>


int ctv(char c){
    if(c>='0' && c<='9') return c-'0';
    if(c>='a' && c<='z') return c-'a'+10;
    if(c>='A' && c<='Z') return c-'A'+10;
    return -1;
}

long long todecimal( const char *string, int base)
{
    long long result = 0;
    for (int i = 0;string[i] !='\0'; i++){
        int digit = ctv(string[i]);
        result = result * base + digit;
    }
    return result;
}

int main(){
    char num[100];
    int base;

    printf("Enter number: ");
    scanf("%s", num);

    printf("Enter base (<37): ");
    scanf("%d", &base);

    printf("Decimal = %lld\n", todecimal(num, base));

    return 0;
}
