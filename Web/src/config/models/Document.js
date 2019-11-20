const mongoose = require('mongoose');

const documentSchema = mongoose.Schema({
    title: {
        type:String,
        require:true
    },
    content:{
        type:[String],
        require:true,
    },
    imageFile:{
        type:String,
    },
    keyPhrase:{
        type:[String],
    },
    user:{
        type:mongoose.Schema.Types.ObjectId,
        ref:'User'
    }
})

const document = mongoose.model('Document',documentSchema)

module.exports = document