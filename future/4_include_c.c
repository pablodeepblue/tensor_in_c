float dot_product(float* vector1, float* vector2, int len){
    float out = 0.0;
    for (int i = 0; i < len; i++){
        out += vector1[i] * vector2[i];
    }
    return out;
}