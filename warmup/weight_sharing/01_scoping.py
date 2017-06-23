import tensorflow as tf

def main():
    inp = tf.constant(0)
     
    

    #init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:        
        #sess.run(init_op)
        sess.run(o2)
        print o2.eval()
        
if __name__ == '__main__':
    main()
